package whitedeath.rl;

import arc.*;
import arc.backend.headless.*;
import arc.files.*;
import arc.math.*;
import arc.util.*;
import mindustry.*;
import mindustry.content.*;
import mindustry.core.*;
import mindustry.core.GameState.*;
import mindustry.game.*;
import mindustry.gen.*;
import mindustry.mod.*;
import mindustry.net.*;
import mindustry.type.*;
import mindustry.world.*;
import mindustry.world.blocks.distribution.*;
import mindustry.world.blocks.production.*;

import java.io.*;
import java.nio.charset.*;
import java.util.*;

import static mindustry.Vars.*;

/**
 * Headless Mindustry simulator for reinforcement learning.
 *
 * The game loop is stopped right after initialization (the same trick Mindustry's own tests use),
 * so time only advances when the agent asks for it. That makes the simulation run as fast as the CPU allows.
 *
 * Protocol: one command per line on stdin, one JSON response per line on stdout.
 *   reset <seed>   start a new episode on a freshly generated map
 *   step <action>  apply an action, simulate ticksPerStep game ticks
 *   close          exit
 *
 * Task "mining": the agent places mechanical drills and conveyors to deliver as much copper and lead
 * to the core as possible before the episode ends. Buildings appear instantly and cost core resources.
 */
public class SimServer{
    // Observation channels (one byte per cell each)
    static final int
    chOreCopper = 0, chOreLead = 1, chWall = 2, chCore = 3, chDrill = 4,
    chConveyor = 5, // 5..8: conveyor facing right, up, left, down
    chHasItems = 9,
    channels = 10;

    static int size = 24, ticksPerStep = 60, maxSteps = 150, startCopper = 120;
    /** Weight of the potential-based shaping reward (0 = off) and the discount it is computed with. */
    static float shaping = 0f, gamma = 0.99f;

    static PrintStream out;
    static Rand rand = new Rand();
    static Building core;
    static int stepIndex;
    static long totalIncome;
    static float lastPotential;

    public static void main(String[] args) throws Exception{
        for(int i = 0; i + 1 < args.length; i += 2){
            String value = args[i + 1];
            switch(args[i]){
                case "--size" -> size = Integer.parseInt(value);
                case "--ticks" -> ticksPerStep = Integer.parseInt(value);
                case "--steps" -> maxSteps = Integer.parseInt(value);
                case "--copper" -> startCopper = Integer.parseInt(value);
                case "--shaping" -> shaping = Float.parseFloat(value);
                case "--gamma" -> gamma = Float.parseFloat(value);
                default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
            }
        }

        //stdout is reserved for the protocol; game logs go to stderr
        out = new PrintStream(new FileOutputStream(FileDescriptor.out), false, StandardCharsets.UTF_8);
        System.setOut(System.err);

        launch();

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while((line = in.readLine()) != null){
            String[] cmd = line.trim().split(" ");
            switch(cmd[0]){
                case "reset" -> reset(cmd.length > 1 ? Long.parseLong(cmd[1]) : System.nanoTime());
                case "step" -> step(Integer.parseInt(cmd[1]));
                case "close" -> System.exit(0);
                default -> respondError("unknown command: " + cmd[0]);
            }
        }
        System.exit(0);
    }

    static void launch() throws Exception{
        boolean[] begins = {false};
        Throwable[] error = {null};
        Fi dataDir = new Fi(System.getProperty("java.io.tmpdir")).child("whitedeath-rl-" + ProcessHandle.current().pid());
        Log.useColors = false;

        ApplicationCore app = new ApplicationCore(){
            @Override
            public void setup(){
                Core.settings.setDataDirectory(dataDir);
                headless = true;
                net = new Net(null);
                tree = new FileTree();
                Vars.init();
                world = new World(){
                    @Override
                    public float getDarkness(int x, int y){
                        return 0;
                    }
                };
                content.createBaseContent();
                mods.loadScripts();
                content.createModContent();

                add(logic = new Logic());
                add(netServer = new NetServer());

                content.init();
                mods.eachClass(Mod::init);
            }

            @Override
            public void init(){
                super.init();
                begins[0] = true;
                //stop the application loop: from here on the main thread drives the simulation
                Thread.currentThread().interrupt();
            }
        };

        new HeadlessApplication(app, t -> {
            if(!begins[0]) error[0] = t;
        });

        while(!begins[0]){
            if(error[0] != null) throw new RuntimeException(error[0]);
            Thread.sleep(10);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(dataDir::deleteDirectory));
        Time.setDeltaProvider(() -> 1f);
    }

    static void reset(long seed){
        rand.setSeed(seed);
        logic.reset();
        world.loadGenerator(size, size, SimServer::generate);

        state.rules = new Rules();
        state.rules.waves = false;
        state.rules.fog = false;
        state.rules.limitMapArea = false;
        state.set(State.playing);

        core = Team.sharded.core();
        core.items.clear();
        core.items.set(Items.copper, startCopper);

        stepIndex = 0;
        totalIncome = 0;
        lastPotential = potential();
        respond(0f, 0, false);
    }

    static void generate(Tiles tiles){
        for(int x = 0; x < size; x++){
            for(int y = 0; y < size; y++){
                boolean edge = x == 0 || y == 0 || x == size - 1 || y == size - 1;
                tiles.set(x, y, new Tile(x, y, Blocks.stone, Blocks.air, edge ? Blocks.stoneWall : Blocks.air));
            }
        }

        //rock clusters
        int rocks = rand.random(2, 5);
        for(int i = 0; i < rocks; i++){
            blob(tiles, rand.random(2, size - 3), rand.random(2, size - 3), rand.random(1f, 2.2f), t -> t.setBlock(Blocks.stoneWall));
        }

        //ore patches, alternating copper and lead so both always exist
        int patches = rand.random(3, 5);
        for(int i = 0; i < patches; i++){
            Block ore = i % 2 == 0 ? Blocks.oreCopper : Blocks.oreLead;
            blob(tiles, rand.random(2, size - 3), rand.random(2, size - 3), rand.random(1.5f, 2.8f), t -> {
                if(!t.solid()) t.setOverlay(ore);
            });
        }

        //core with a clear ring around it
        int cx = rand.random(4, size - 5), cy = rand.random(4, size - 5);
        for(int x = cx - 2; x <= cx + 2; x++){
            for(int y = cy - 2; y <= cy + 2; y++){
                Tile t = tiles.get(x, y);
                t.setBlock(Blocks.air);
                t.setOverlay(Blocks.air);
            }
        }
        tiles.get(cx, cy).setBlock(Blocks.coreShard, Team.sharded, 0);
    }

    static void blob(Tiles tiles, int cx, int cy, float radius, arc.func.Cons<Tile> cons){
        int r = Mathf.ceil(radius);
        for(int x = cx - r; x <= cx + r; x++){
            for(int y = cy - r; y <= cy + r; y++){
                Tile t = tiles.get(x, y);
                if(t == null || x <= 0 || y <= 0 || x >= size - 1 || y >= size - 1) continue;
                if(Mathf.dst(x, y, cx, cy) <= radius && rand.chance(0.85f)) cons.get(t);
            }
        }
    }

    // ---- actions ----
    // 0: do nothing; 1..S²: drill at cell; S²+1..5S²: conveyor at cell with rotation (cell * 4 + rotation)

    static int actionCount(){
        return 1 + size * size * 5;
    }

    static void step(int action){
        if(action < 0 || action >= actionCount()){
            respondError("action out of range: " + action);
            return;
        }

        boolean placed = false;
        if(action > 0){
            int a = action - 1, cells = size * size;
            Block block;
            int cell, rotation;
            if(a < cells){
                block = Blocks.mechanicalDrill;
                cell = a;
                rotation = 0;
            }else{
                block = Blocks.conveyor;
                cell = (a - cells) / 4;
                rotation = (a - cells) % 4;
            }
            placed = place(block, cell % size, cell / size, rotation);
        }

        int before = coreItems();
        for(int i = 0; i < ticksPerStep && core.isValid(); i++){
            Time.update();
            logic.update();
        }
        int income = Math.max(coreItems() - before, 0);
        totalIncome += income;
        stepIndex++;

        float reward = income * 0.1f;
        float potential = potential();
        reward += shaping * (gamma * potential - lastPotential);
        lastPotential = potential;
        respond(reward, income, stepIndex >= maxSteps || !core.isValid());
        if(action > 0 && !placed) Log.warn("Invalid action @ was ignored", action);
    }

    static boolean canPlace(Block block, int x, int y, int rotation){
        return core.items.has(block.requirements) && Build.validPlace(block, Team.sharded, x, y, rotation, false);
    }

    static boolean place(Block block, int x, int y, int rotation){
        if(!canPlace(block, x, y, rotation)) return false;
        core.items.remove(block.requirements);
        Tile tile = world.tile(x, y);
        tile.setBlock(block, Team.sharded, rotation);
        if(tile.build != null) tile.build.placed();
        return true;
    }

    static int coreItems(){
        return core.items.get(Items.copper) + core.items.get(Items.lead);
    }

    // ---- reward shaping ----

    /**
     * Potential of the current base: for every drill, how close its ore gets to the core
     * (following the conveyor chain it feeds), weighted by how much ore it covers.
     * A drill whose chain reaches the core scores its full weight; an unconnected one at most half,
     * growing as its chain gets closer. Rewarding the change of this potential (gamma * new - old)
     * guides the agent without changing which strategy is optimal.
     */
    static float potential(){
        int[] dist = coreDistances();
        float maxDist = size / 2f, total = 0f;
        for(Building b : buildings()){
            if(!(b instanceof Drill.DrillBuild drill) || drill.dominantItems == 0) continue;

            float best = maxDist;
            for(Tile t : drill.tile.getLinkedTiles(new arc.struct.Seq<>())) best = Math.min(best, dist(dist, t));
            for(Building p : neighbors(drill)){
                if(p == core) best = 0;
                else if(p instanceof Conveyor.ConveyorBuild && p.front() != drill) best = Math.min(best, followChain(p, dist));
            }
            float progress = best == 0 ? 1f : 0.5f * (1f - Math.min(best, maxDist) / maxDist);
            total += drill.dominantItems / 4f * progress;
        }
        return total;
    }

    /** Closest distance to the core that items on this conveyor chain reach; 0 if they arrive. */
    static float followChain(Building start, int[] dist){
        float best = Float.MAX_VALUE;
        Set<Building> visited = new HashSet<>();
        Building cur = start;
        while(cur instanceof Conveyor.ConveyorBuild && visited.add(cur)){
            best = Math.min(best, dist(dist, cur.tile));
            Building next = cur.front();
            if(next == core) return 0;
            if(next == null || next.front() == cur) break;
            cur = next;
        }
        return best;
    }

    static float dist(int[] dist, Tile t){
        int d = dist[t.y * size + t.x];
        return d < 0 ? Float.MAX_VALUE : d;
    }

    /** Walking distance from every cell to the core, ignoring buildings; -1 if unreachable. */
    static int[] coreDistances(){
        int[] dist = new int[size * size];
        Arrays.fill(dist, -1);
        ArrayDeque<Tile> queue = new ArrayDeque<>();
        for(Tile t : core.tile.getLinkedTiles(new arc.struct.Seq<>())){
            dist[t.y * size + t.x] = 0;
            queue.add(t);
        }
        while(!queue.isEmpty()){
            Tile t = queue.poll();
            for(int r = 0; r < 4; r++){
                Tile n = t.nearby(r);
                if(n == null || n.block().isStatic() || dist[n.y * size + n.x] >= 0) continue;
                dist[n.y * size + n.x] = dist[t.y * size + t.x] + 1;
                queue.add(n);
            }
        }
        return dist;
    }

    // ---- scripted expert ----
    // Decides from the current state only (no plan kept between steps), so it can label any state the
    // learning agent visits, including its mistakes. That is what DAgger imitation learning needs.

    /** Conveyors needed from a free cell to deliver into the core (0 = unreachable) and the direction to go. */
    static int[] exitDist;
    static byte[] exitDir;

    static int expertAction(){
        Set<Building> connected = connectedConveyors();
        computeExitField(connected);
        int copper = core.items.get(Items.copper), cells = size * size;
        arc.struct.Seq<Tile> footprint = new arc.struct.Seq<>();

        //1. lay the next conveyor for the unconnected drill closest to being connected
        int bestStart = -1, bestLength = Integer.MAX_VALUE;
        for(Building b : buildings()){
            if(!(b instanceof Drill.DrillBuild drill)) continue;
            boolean done = false;
            int start = -1, length = Integer.MAX_VALUE;
            for(Building p : neighbors(drill)){
                if(p == core || connected.contains(p) && p.front() != drill){
                    done = true;
                    break;
                }
                if(p instanceof Conveyor.ConveyorBuild && p.front() != drill){
                    Building end = walkChain(p);
                    Tile front = end.tile.nearby(end.rotation);
                    if(isFree(front) && exitDist[index(front)] > 0 && exitDist[index(front)] < length){
                        start = index(front);
                        length = exitDist[start];
                    }
                }
            }
            if(done) continue;
            drill.tile.getLinkedTiles(footprint);
            for(Tile t : footprint){
                for(int r = 0; r < 4; r++){
                    Tile n = t.nearby(r);
                    if(isFree(n) && exitDist[index(n)] > 0 && exitDist[index(n)] < length){
                        start = index(n);
                        length = exitDist[start];
                    }
                }
            }
            if(start >= 0 && length < bestLength){
                bestStart = start;
                bestLength = length;
            }
        }
        if(bestStart >= 0){
            int x = bestStart % size, y = bestStart / size, rotation = exitDir[bestStart];
            return copper >= 1 && canPlace(Blocks.conveyor, x, y, rotation) ? 1 + cells + bestStart * 4 + rotation : 0;
        }

        //2. every drill is connected: place a new one on the richest ore that is cheap to connect
        int bestCell = -1, bestScore = Integer.MIN_VALUE;
        for(int cell = 0; cell < cells; cell++){
            int x = cell % size, y = cell / size;
            if(!canPlace(Blocks.mechanicalDrill, x, y, 0)) continue;
            int ores = 0, length = Integer.MAX_VALUE;
            for(int dx = 0; dx < 2; dx++){
                for(int dy = 0; dy < 2; dy++){
                    Block ore = world.tile(x + dx, y + dy).overlay();
                    if(ore == Blocks.oreCopper || ore == Blocks.oreLead) ores++;
                }
            }
            if(ores < 2) continue;
            for(int dx = -1; dx <= 2; dx++){
                for(int dy = -1; dy <= 2; dy++){
                    boolean inside = dx >= 0 && dx < 2 && dy >= 0 && dy < 2, corner = (dx == -1 || dx == 2) && (dy == -1 || dy == 2);
                    if(inside || corner) continue;
                    Tile n = world.tile(x + dx, y + dy);
                    if(n == null) continue;
                    if(n.build == core || connected.contains(n.build) && !facesFootprint(n.build, x, y)) length = 0;
                    else if(isFree(n) && exitDist[index(n)] > 0) length = Math.min(length, exitDist[index(n)]);
                }
            }
            if(length == Integer.MAX_VALUE || copper < Blocks.mechanicalDrill.requirements[0].amount + length) continue;
            int score = ores * 10 - length;
            if(score > bestScore){
                bestScore = score;
                bestCell = cell;
            }
        }
        return bestCell >= 0 ? 1 + bestCell : 0;
    }

    static boolean facesFootprint(Building conveyor, int x, int y){
        Tile f = conveyor.tile.nearby(conveyor.rotation);
        return f != null && f.x >= x && f.x <= x + 1 && f.y >= y && f.y <= y + 1;
    }

    /** Free cells: BFS outward from the cells that can deliver straight into the core or a connected conveyor. */
    static void computeExitField(Set<Building> connected){
        exitDist = new int[size * size];
        exitDir = new byte[size * size];
        ArrayDeque<Tile> queue = new ArrayDeque<>();
        for(int i = 0; i < size * size; i++){
            Tile t = world.tile(i % size, i / size);
            if(!isFree(t)) continue;
            for(int r = 0; r < 4; r++){
                Tile n = t.nearby(r);
                if(n == null || n.build == null) continue;
                boolean intoConveyor = connected.contains(n.build) && n.build.tile.nearby(n.build.rotation) != t;
                if(n.build == core || intoConveyor){
                    exitDist[i] = 1;
                    exitDir[i] = (byte)r;
                    queue.add(t);
                    break;
                }
            }
        }
        while(!queue.isEmpty()){
            Tile t = queue.poll();
            for(int r = 0; r < 4; r++){
                Tile n = t.nearby(r);
                if(!isFree(n) || exitDist[index(n)] > 0) continue;
                exitDist[index(n)] = exitDist[index(t)] + 1;
                exitDir[index(n)] = (byte)((r + 2) % 4);
                queue.add(n);
            }
        }
    }

    static Set<Building> connectedConveyors(){
        Set<Building> result = new HashSet<>();
        for(Building b : buildings()){
            if(b instanceof Conveyor.ConveyorBuild && walkChain(b) == core) result.add(b);
        }
        return result;
    }

    /** Follows a conveyor chain; returns the core if items arrive there, otherwise the last conveyor. */
    static Building walkChain(Building start){
        Set<Building> visited = new HashSet<>();
        Building cur = start;
        while(visited.add(cur)){
            Building next = cur.front();
            if(next == core) return core;
            if(!(next instanceof Conveyor.ConveyorBuild) || next.front() == cur) return cur;
            cur = next;
        }
        return cur;
    }

    /**
     * All buildings, read from the tiles. Groups.build and Building.proximity are not reliable here:
     * with the game loop driven by hand, Groups.build misses freshly placed buildings and proximity lags a step.
     */
    static arc.struct.Seq<Building> buildings(){
        arc.struct.Seq<Building> result = new arc.struct.Seq<>();
        for(Tile t : world.tiles){
            if(t.build != null && t.isCenter()) result.add(t.build);
        }
        return result;
    }

    static Set<Building> neighbors(Building b){
        Set<Building> result = new LinkedHashSet<>();
        for(Tile t : b.tile.getLinkedTiles(new arc.struct.Seq<>())){
            for(int r = 0; r < 4; r++){
                Tile n = t.nearby(r);
                if(n != null && n.build != null && n.build != b) result.add(n.build);
            }
        }
        return result;
    }

    static boolean isFree(Tile t){
        return t != null && t.build == null && t.block() == Blocks.air && !t.floor().isDeep();
    }

    static int index(Tile t){
        return t.y * size + t.x;
    }

    // ---- observation ----

    static byte[] observe(){
        byte[] obs = new byte[channels * size * size];
        for(int y = 0; y < size; y++){
            for(int x = 0; x < size; x++){
                Tile t = world.tile(x, y);
                int cell = y * size + x;
                Block ore = t.overlay();
                if(ore == Blocks.oreCopper) obs[chOreCopper * size * size + cell] = 1;
                if(ore == Blocks.oreLead) obs[chOreLead * size * size + cell] = 1;
                if(t.block().isStatic()) obs[chWall * size * size + cell] = 1;

                Building b = t.build;
                if(b == null) continue;
                if(b.block == Blocks.coreShard) obs[chCore * size * size + cell] = 1;
                if(b.block instanceof Drill) obs[chDrill * size * size + cell] = 1;
                if(b.block instanceof Conveyor) obs[(chConveyor + b.rotation) * size * size + cell] = 1;
                if(b.block != Blocks.coreShard && b.items != null && b.items.total() > 0) obs[chHasItems * size * size + cell] = 1;
            }
        }
        return obs;
    }

    static byte[] actionMask(){
        int cells = size * size;
        byte[] mask = new byte[actionCount()];
        mask[0] = 1;
        boolean drillAffordable = core.items.has(Blocks.mechanicalDrill.requirements);
        boolean conveyorAffordable = core.items.has(Blocks.conveyor.requirements);
        for(int cell = 0; cell < cells; cell++){
            int x = cell % size, y = cell / size;
            if(drillAffordable && Build.validPlace(Blocks.mechanicalDrill, Team.sharded, x, y, 0, false)){
                mask[1 + cell] = 1;
            }
            if(conveyorAffordable && Build.validPlace(Blocks.conveyor, Team.sharded, x, y, 0, false)){
                for(int r = 0; r < 4; r++) mask[1 + cells + cell * 4 + r] = 1;
            }
        }
        return mask;
    }

    // ---- protocol ----

    static void respond(float reward, int income, boolean done){
        Base64.Encoder b64 = Base64.getEncoder();
        String json = "{\"obs\":\"" + b64.encodeToString(observe()) +
            "\",\"mask\":\"" + b64.encodeToString(actionMask()) +
            "\",\"reward\":" + reward +
            ",\"done\":" + done +
            ",\"copper\":" + core.items.get(Items.copper) +
            ",\"lead\":" + core.items.get(Items.lead) +
            ",\"income\":" + income +
            ",\"total_income\":" + totalIncome +
            ",\"step\":" + stepIndex +
            ",\"size\":" + size +
            ",\"channels\":" + channels +
            ",\"max_steps\":" + maxSteps +
            ",\"potential\":" + lastPotential +
            ",\"expert\":" + expertAction() + "}";
        out.println(json);
        out.flush();
    }

    static void respondError(String message){
        out.println("{\"error\":\"" + message.replace("\"", "'") + "\"}");
        out.flush();
    }
}
