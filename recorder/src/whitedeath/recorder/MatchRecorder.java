package whitedeath.recorder;

import arc.*;
import arc.files.*;
import arc.func.*;
import arc.struct.*;
import arc.util.*;
import mindustry.ai.*;
import mindustry.ai.types.*;
import mindustry.core.*;
import mindustry.ctype.*;
import mindustry.game.*;
import mindustry.game.EventType.*;
import mindustry.game.Teams.*;
import mindustry.gen.*;
import mindustry.io.*;
import mindustry.type.*;
import mindustry.world.*;
import mindustry.world.blocks.ConstructBlock.*;

import java.io.*;
import java.nio.*;
import java.nio.charset.*;
import java.security.*;
import java.text.*;
import java.util.*;
import java.util.zip.*;

import static mindustry.Vars.*;

/**
 * Passively records the match the client is watching or playing: it only reads the world state the client
 * already receives and never sends anything to the server.
 *
 * Output, one folder per match:
 *   events.jsonl.gz      one JSON object per line, "t" = game ticks since the match started (60 per second)
 *   keyframe-TICK.msav   full world snapshots, loadable with SaveIO (and by the RL simulator)
 * Player names are replaced by salted hashes, stable across matches so players can be rated later.
 */
public class MatchRecorder{
    public static final int formatVersion = 1;

    public int snapshotTicks = 30, playerTicks = 300, keyframeTicks = 3600;

    final Fi root;
    final String salt;

    Fi dir;
    Writer out;
    boolean active;
    float time;
    int nextSnapshot, nextPlayers, nextKeyframe, keyframes;
    int width, height;
    int[] buildings;

    public MatchRecorder(Fi root, String salt){
        this.root = root;
        this.salt = salt;
    }

    /** Wires the recorder to game events. shouldRecord decides, when a map loads, whether to record it. */
    public void register(Boolp shouldRecord){
        Events.on(WorldLoadEvent.class, e -> {
            stop("next-map", null);
            if(shouldRecord.get()) start();
        });
        Events.on(GameOverEvent.class, e -> stop("game-over", e.winner));
        Events.on(ResetEvent.class, e -> stop("left", null));
        Events.on(BlockBuildEndEvent.class, this::onBuildEnd);
        Events.on(BlockDestroyEvent.class, this::onDestroy);
        Events.on(ConfigEvent.class, this::onConfig);
        Events.run(Trigger.update, this::update);
    }

    public boolean active(){
        return active;
    }

    public float seconds(){
        return time / 60f;
    }

    public Fi folder(){
        return dir;
    }

    // ---- lifecycle ----

    public void start(){
        try{
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date());
            String base = stamp + "-" + sanitize(state.map == null ? "map" : state.map.name());
            dir = root.child(base);
            for(int i = 2; dir.exists(); i++) dir = root.child(base + "-" + i);
            dir.mkdirs();
            out = new BufferedWriter(new OutputStreamWriter(
                new GZIPOutputStream(new FileOutputStream(dir.child("events.jsonl.gz").file()), 1 << 16, true),
                StandardCharsets.UTF_8));
        }catch(IOException e){
            Log.err("[WhiteDeath] Could not start recording", e);
            return;
        }

        active = true;
        time = 0;
        nextSnapshot = nextPlayers = nextKeyframe = 0;
        keyframes = 0;
        width = world.width();
        height = world.height();
        buildings = new int[width * height];

        writeStart();
        Log.info("[WhiteDeath] Recording match to @", dir.absolutePath());
    }

    public void stop(String reason, Team winner){
        if(!active) return;
        Json line = line("end").str("reason", reason).str("winner", winner == null ? null : winner.name);
        line.raw("teams", teamsJson());
        write(line);
        try{
            out.close();
        }catch(IOException e){
            Log.err("[WhiteDeath] Could not finish recording", e);
        }
        active = false;
        out = null;
        Log.info("[WhiteDeath] Recording finished (@, @s): @", reason, (int)seconds(), dir.absolutePath());
    }

    public void update(){
        if(!active || !state.isGame()) return;
        if(world.width() != width || world.height() != height){
            stop("map-changed", null);
            return;
        }
        if(!state.isPaused()) time += Time.delta;
        int tick = (int)time;

        if(tick >= nextSnapshot){
            nextSnapshot = tick + snapshotTicks;
            snapshotBuildings();
            snapshotUnits();
        }
        if(tick >= nextPlayers){
            nextPlayers = tick + playerTicks;
            snapshotPlayers();
            snapshotCores();
        }
        if(tick >= nextKeyframe){
            nextKeyframe = tick + keyframeTicks;
            keyframe();
        }
    }

    // ---- snapshots ----

    void writeStart(){
        Json line = line("start")
            .num("format", formatVersion)
            .num("game", Version.build)
            .str("map", state.map == null ? null : state.map.name())
            .num("w", width).num("h", height)
            .str("recorded", new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssZ").format(new Date()))
            .str("server", net.client() ? "remote" : "local");

        Json rules = new Json()
            .bool("pvp", state.rules.pvp).bool("fog", state.rules.fog).bool("attack", state.rules.attackMode)
            .num("unitCap", state.rules.unitCap).num("buildSpeed", state.rules.buildSpeedMultiplier)
            .num("unitBuildSpeed", state.rules.unitBuildSpeedMultiplier).num("buildCost", state.rules.buildCostMultiplier);
        line.raw("rules", rules.close());
        line.raw("teams", teamsJson());

        line.raw("blocks", names(content.blocks()));
        line.raw("units", names(content.units()));
        line.raw("items", names(content.items()));
        StringBuilder commands = new StringBuilder("[");
        for(int i = 0; i < UnitCommand.all.size; i++){
            if(i > 0) commands.append(',');
            Json.quote(commands, UnitCommand.all.get(i).name);
        }
        line.raw("commands", commands.append(']').toString());

        //terrain as little-endian uint16 block ids, row-major from y = 0
        short[][] layers = new short[3][width * height];
        for(Tile t : world.tiles){
            int i = t.y * width + t.x;
            layers[0][i] = t.floor().id;
            layers[1][i] = t.overlay().id;
            layers[2][i] = t.block().isStatic() ? t.block().id : 0;
        }
        Json terrain = new Json();
        String[] names = {"floor", "overlay", "wall"};
        for(int l = 0; l < 3; l++){
            ByteBuffer buf = ByteBuffer.allocate(width * height * 2).order(ByteOrder.LITTLE_ENDIAN);
            for(short s : layers[l]) buf.putShort(s);
            terrain.str(names[l], Base64.getEncoder().encodeToString(buf.array()));
        }
        line.raw("terrain", terrain.close());
        write(line);
    }

    String teamsJson(){
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for(TeamData data : state.teams.getActive()){
            if(!first) sb.append(',');
            first = false;
            sb.append("{\"id\":").append(data.team.id).append(",\"name\":");
            Json.quote(sb, data.team.name);
            sb.append(",\"cores\":[");
            for(int i = 0; i < data.cores.size; i++){
                if(i > 0) sb.append(',');
                sb.append('[').append(data.cores.get(i).tile.x).append(',').append(data.cores.get(i).tile.y).append(']');
            }
            sb.append("]}");
        }
        return sb.append(']').toString();
    }

    /** Buildings that appeared, disappeared or changed since the last snapshot: [x, y, block, team, rotation, site]. */
    void snapshotBuildings(){
        StringBuilder changes = new StringBuilder("[");
        int count = 0;
        for(Tile t : world.tiles){
            int i = t.y * width + t.x, packed = 0;
            Building b = t.build;
            if(b != null && t.isCenter()){
                if(b instanceof ConstructBuild site){
                    Block planned = site.current != null ? site.current : site.previous;
                    packed = pack(planned == null ? b.block.id : planned.id, b.team.id, b.rotation, true);
                }else{
                    packed = pack(b.block.id, b.team.id, b.rotation, false);
                }
            }
            if(packed == buildings[i]) continue;
            buildings[i] = packed;
            if(count++ > 0) changes.append(',');
            changes.append('[').append(t.x).append(',').append(t.y).append(',');
            if(packed == 0){
                changes.append("-1,0,0,0]");
            }else{
                changes.append((packed >>> 12) - 1).append(',').append((packed >>> 4) & 0xff).append(',')
                    .append((packed >>> 1) & 3).append(',').append(packed & 1).append(']');
            }
        }
        if(count > 0) write(line("blocks").raw("c", changes.append(']').toString()));
    }

    static int pack(int block, int team, int rotation, boolean site){
        return (block + 1) << 12 | (team & 0xff) << 4 | (rotation & 3) << 1 | (site ? 1 : 0);
    }

    /** Every unit: [id, type, team, x*10, y*10 (tiles), health %, command, targetX*10, targetY*10, player]. */
    void snapshotUnits(){
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for(Unit u : Groups.unit){
            if(!first) sb.append(',');
            first = false;
            int cmd = -1, tx = -1, ty = -1;
            if(u.controller() instanceof CommandAI ai){
                if(ai.command != null) cmd = ai.command.id;
                if(ai.targetPos != null){
                    tx = Math.round(ai.targetPos.x / tilesize * 10);
                    ty = Math.round(ai.targetPos.y / tilesize * 10);
                }
            }
            sb.append('[').append(u.id).append(',').append(u.type.id).append(',').append(u.team.id).append(',')
                .append(Math.round(u.x / tilesize * 10)).append(',').append(Math.round(u.y / tilesize * 10)).append(',')
                .append(Math.round(u.health / Math.max(u.maxHealth, 1f) * 100)).append(',')
                .append(cmd).append(',').append(tx).append(',').append(ty).append(',');
            Player p = u.getPlayer();
            if(p == null) sb.append("null");
            else Json.quote(sb, pseudonym(p.name));
            sb.append(']');
        }
        write(line("units").raw("u", sb.append(']').toString()));
    }

    void snapshotPlayers(){
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for(Player p : Groups.player){
            if(!first) sb.append(',');
            first = false;
            Json o = new Json().str("p", pseudonym(p.name)).num("team", p.team().id)
                .num("x", Math.round(p.x / tilesize * 10) / 10f).num("y", Math.round(p.y / tilesize * 10) / 10f)
                .num("unit", p.unit() == null || p.unit().type == null ? -1 : p.unit().type.id)
                .bool("me", p == player);
            sb.append(o.close());
        }
        write(line("players").raw("p", sb.append(']').toString()));
    }

    /** Core inventories as the client sees them (the enemy's may not be synced by the server). */
    void snapshotCores(){
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for(TeamData data : state.teams.getActive()){
            if(data.core() == null) continue;
            if(!first) sb.append(',');
            first = false;
            Json items = new Json();
            for(Item item : content.items()){
                int amount = data.core().items.get(item);
                if(amount > 0) items.num(item.name, amount);
            }
            sb.append("{\"team\":").append(data.team.id).append(",\"cores\":").append(data.cores.size)
                .append(",\"items\":").append(items.close()).append('}');
        }
        write(line("cores").raw("c", sb.append(']').toString()));
    }

    void keyframe(){
        int tick = (int)time;
        try{
            SaveIO.write(dir.child("keyframe-" + tick + ".msav"));
            keyframes++;
            write(line("keyframe").str("file", "keyframe-" + tick + ".msav"));
            out.flush();
        }catch(Throwable e){
            Log.err("[WhiteDeath] Could not write keyframe", e);
        }
    }

    // ---- events ----

    void onBuildEnd(BlockBuildEndEvent e){
        if(!active) return;
        Block block = e.breaking ? null : e.tile.block();
        write(line("built").num("x", e.tile.x).num("y", e.tile.y)
            .num("block", block == null ? -1 : block.id).num("team", e.team.id)
            .num("rot", e.tile.build == null ? 0 : e.tile.build.rotation).bool("breaking", e.breaking)
            .str("by", builder(e.unit)).num("unit", e.unit == null ? -1 : e.unit.type.id));
    }

    void onDestroy(BlockDestroyEvent e){
        if(!active) return;
        write(line("destroyed").num("x", e.tile.x).num("y", e.tile.y).num("block", e.tile.block().id).num("team", e.tile.team().id));
    }

    void onConfig(ConfigEvent e){
        if(!active || e.tile == null) return;
        Object v = e.value;
        String value = v == null ? null : v instanceof MappableContent c ? c.name : String.valueOf(v);
        if(value != null && value.length() > 64) value = value.substring(0, 64);
        write(line("config").num("x", e.tile.tile.x).num("y", e.tile.tile.y).num("block", e.tile.block.id)
            .num("team", e.tile.team.id).str("value", value).str("by", e.player == null ? null : pseudonym(e.player.name)));
    }

    String builder(Unit unit){
        if(unit == null) return null;
        Player p = unit.getPlayer();
        return p == null ? null : pseudonym(p.name);
    }

    // ---- helpers ----

    /** Stable, non-reversible id for a player name (colors stripped), so ratings can be computed without names. */
    public String pseudonym(String name){
        try{
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest((salt + "|" + Strings.stripColors(name).trim()).getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for(int i = 0; i < 5; i++) sb.append(String.format("%02x", hash[i]));
            return sb.toString();
        }catch(NoSuchAlgorithmException e){
            throw new RuntimeException(e);
        }
    }

    Json line(String type){
        return new Json().num("t", (int)time).str("e", type);
    }

    void write(Json json){
        if(out == null) return;
        try{
            out.write(json.close());
            out.write('\n');
        }catch(IOException e){
            Log.err("[WhiteDeath] Write failed, recording stopped", e);
            active = false;
        }
    }

    static String names(Seq<? extends MappableContent> list){
        StringBuilder sb = new StringBuilder("[");
        for(int i = 0; i < list.size; i++){
            if(i > 0) sb.append(',');
            Json.quote(sb, list.get(i).name);
        }
        return sb.append(']').toString();
    }

    static String sanitize(String name){
        String clean = Strings.stripColors(name).replaceAll("[^A-Za-z0-9А-Яа-я_-]+", "_");
        return clean.length() > 40 ? clean.substring(0, 40) : clean;
    }

    /** Minimal JSON object builder; keys are trusted constants, string values are escaped. */
    static class Json{
        final StringBuilder sb = new StringBuilder("{");

        Json key(String key){
            if(sb.length() > 1) sb.append(',');
            sb.append('"').append(key).append("\":");
            return this;
        }

        Json num(String key, double value){
            key(key);
            if(value == Math.rint(value) && Math.abs(value) < 1e15) sb.append((long)value);
            else sb.append(value);
            return this;
        }

        Json str(String key, String value){
            key(key);
            if(value == null) sb.append("null");
            else quote(sb, value);
            return this;
        }

        Json bool(String key, boolean value){
            key(key).sb.append(value);
            return this;
        }

        Json raw(String key, String json){
            key(key).sb.append(json);
            return this;
        }

        String close(){
            return sb + "}";
        }

        static void quote(StringBuilder sb, String s){
            sb.append('"');
            for(int i = 0; i < s.length(); i++){
                char c = s.charAt(i);
                switch(c){
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    default -> {
                        if(c < 0x20) sb.append(String.format("\\u%04x", (int)c));
                        else sb.append(c);
                    }
                }
            }
            sb.append('"');
        }
    }
}
