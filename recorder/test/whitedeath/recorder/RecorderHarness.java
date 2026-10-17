package whitedeath.recorder;

import arc.*;
import arc.backend.headless.*;
import arc.files.*;
import arc.math.geom.*;
import arc.util.*;
import mindustry.*;
import mindustry.ai.*;
import mindustry.ai.types.*;
import mindustry.content.*;
import mindustry.core.*;
import mindustry.core.GameState.*;
import mindustry.game.*;
import mindustry.game.EventType.*;
import mindustry.gen.*;
import mindustry.io.*;
import mindustry.mod.*;
import mindustry.net.*;
import mindustry.world.*;

import static mindustry.Vars.*;

/**
 * Headless check of the recorder: plays a scripted 1v1 PvP match (buildings, units with commands,
 * a destroyed block, game over), then verifies the keyframe loads back. Usage: RecorderHarness <outputDir>
 */
public class RecorderHarness{
    public static void main(String[] args) throws Exception{
        Fi out = new Fi(args[0]);
        out.deleteDirectory();
        launch();

        MatchRecorder recorder = new MatchRecorder(out, "test-salt");
        recorder.keyframeTicks = 600;
        recorder.register(() -> state.rules.pvp);

        logic.reset();
        state.rules = new Rules();
        state.rules.pvp = true;
        state.rules.waves = false;
        world.loadGenerator(60, 40, tiles -> {
            for(int x = 0; x < 60; x++){
                for(int y = 0; y < 40; y++){
                    boolean edge = x == 0 || y == 0 || x == 59 || y == 39;
                    tiles.set(x, y, new Tile(x, y, Blocks.stone, (x + y) % 7 == 0 ? Blocks.oreCopper : Blocks.air, edge ? Blocks.stoneWall : Blocks.air));
                }
            }
            tiles.get(10, 20).setBlock(Blocks.coreShard, Team.sharded, 0);
            tiles.get(49, 20).setBlock(Blocks.coreShard, Team.crux, 0);
        });
        state.set(State.playing);
        Check.that(recorder.active(), "recording started on PvP map load");

        for(int tick = 0; tick < 1500; tick++){
            if(tick % 120 == 0){
                int i = tick / 120;
                world.tile(14 + i, 16).setBlock(Blocks.conveyor, Team.sharded, 0);
                Events.fire(new BlockBuildEndEvent(world.tile(14 + i, 16), null, Team.sharded, false, null));
                world.tile(45 - i, 24).setBlock(Blocks.duo, Team.crux, 2);
            }
            if(tick == 300){
                Unit dagger = UnitTypes.dagger.spawn(Team.sharded, 14 * tilesize, 22 * tilesize);
                if(dagger.controller() instanceof CommandAI ai){
                    ai.command(UnitCommand.moveCommand);
                    ai.targetPos = new Vec2(40 * tilesize, 22 * tilesize);
                }
                UnitTypes.flare.spawn(Team.crux, 45 * tilesize, 18 * tilesize);
            }
            if(tick == 900) world.tile(45, 24).build.kill();
            Time.update();
            logic.update();
        }
        Events.fire(new GameOverEvent(Team.sharded));
        Check.that(!recorder.active(), "recording stopped on game over");

        Fi match = new arc.struct.Seq<>(out.list()).sort(java.util.Comparator.comparing(Fi::name)).first(); // the reload below records again into a "-2" folder
        Fi[] keyframes = match.findAll(f -> f.extension().equals("msav")).toArray(Fi.class);
        Check.that(keyframes.length >= 2, "keyframes written: " + keyframes.length);

        logic.reset();
        SaveIO.load(keyframes[keyframes.length - 1]);
        Check.that(world.width() == 60 && world.height() == 40, "keyframe loads back with the right size");
        Check.that(state.teams.cores(Team.crux).size == 1 && state.teams.cores(Team.sharded).size == 1, "keyframe has both cores");
        Check.that(world.tile(16, 16).block() == Blocks.conveyor, "keyframe has the placed conveyors");

        System.out.println("MATCH_DIR=" + match.absolutePath());
        System.exit(0);
    }

    static class Check{
        static void that(boolean ok, String what){
            System.out.println((ok ? "PASS " : "FAIL ") + what);
            if(!ok) System.exit(1);
        }
    }

    static void launch() throws Exception{
        boolean[] begins = {false};
        Throwable[] error = {null};
        Fi dataDir = new Fi(System.getProperty("java.io.tmpdir")).child("wd-recorder-test-" + ProcessHandle.current().pid());
        ApplicationCore app = new ApplicationCore(){
            @Override
            public void setup(){
                Core.settings.setDataDirectory(dataDir);
                headless = true;
                net = new Net(null);
                tree = new FileTree();
                Vars.init();
                world = new World();
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
}
