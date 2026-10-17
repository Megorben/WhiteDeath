package whitedeath.recorder;

import arc.*;
import arc.scene.ui.*;
import arc.util.*;
import mindustry.game.EventType.*;
import mindustry.gen.*;
import mindustry.mod.*;
import mindustry.ui.*;

import static mindustry.Vars.*;

/** Client mod: records watched or played PvP matches for imitation learning. Never sends anything to the server. */
public class RecorderMod extends Mod{
    public static MatchRecorder recorder;

    public RecorderMod(){
        if(!Core.settings.has("wd-rec-salt")){
            Core.settings.put("wd-rec-salt", Long.toHexString(new java.security.SecureRandom().nextLong()));
        }
        recorder = new MatchRecorder(dataDirectory.child("whitedeath-recordings"), Core.settings.getString("wd-rec-salt"));
        recorder.register(() -> enabled() && net.client() && (state.rules.pvp || !Core.settings.getBool("wd-rec-pvp-only", true)));
        Events.run(Trigger.update, () -> recorder.keyframeTicks = Core.settings.getInt("wd-rec-keyframe", 60) * 60);
    }

    static boolean enabled(){
        return Core.settings.getBool("wd-rec-enabled", true);
    }

    /**
     * The HUD button. Recording: stop, save and switch automatic recording off.
     * Otherwise: switch automatic recording on and start recording the current game right away,
     * whatever kind of game it is, since the player asked for it explicitly.
     */
    static void toggle(){
        if(recorder.active()){
            recorder.stop("manual", null);
            Core.settings.put("wd-rec-enabled", false);
            ui.showInfoToast(Core.bundle.get("wd-rec.toast.stopped"), 3f);
        }else{
            Core.settings.put("wd-rec-enabled", true);
            if(state.isGame()){
                recorder.start(true);
                ui.showInfoToast(Core.bundle.get("wd-rec.toast.started"), 3f);
            }else{
                ui.showInfoToast(Core.bundle.get("wd-rec.toast.armed"), 3f);
            }
        }
    }

    static String statusText(){
        if(recorder.active()){
            long b = recorder.bytes();
            String size = b < 1024 * 1024 ? (b / 1024) + " KB" : Strings.fixed(b / 1024f / 1024f, 1) + " MB";
            return "[scarlet]" + Iconc.eye + "[] " + Core.bundle.get("wd-rec.button.rec") + "  " + formatTime(recorder.seconds())
                + "\n[lightgray]" + size;
        }
        if(enabled()) return "[accent]" + Iconc.eye + "[] " + Core.bundle.get("wd-rec.button.waiting");
        return "[gray]" + Iconc.eyeOff + "[] " + Core.bundle.get("wd-rec.button.off");
    }

    @Override
    public void init(){
        if(headless) return;

        ui.settings.addCategory("@wd-rec.category", Icon.save, t -> {
            t.checkPref("wd-rec-enabled", true);
            t.checkPref("wd-rec-pvp-only", true);
            t.checkPref("wd-rec-indicator", true);
            t.sliderPref("wd-rec-keyframe", 60, 15, 300, 15, v -> v + " s");
        });

        //left edge, a bit above the middle: the exact middle is where the game shows its tutorial hints
        ui.hudGroup.fill(t -> {
            t.left();
            t.visible(() -> ui.hudfrag.shown && state.isGame() && Core.settings.getBool("wd-rec-indicator", true));
            TextButton button = new TextButton("", Styles.flatBordert);
            button.getLabel().setAlignment(Align.left);
            button.getLabel().setWrap(false);
            button.update(() -> button.setText(statusText()));
            button.clicked(RecorderMod::toggle);
            t.add(button).width(200f).height(58f).padBottom(220f);
        });
    }

    static String formatTime(float seconds){
        int s = (int)seconds;
        return (s / 60) + ":" + (s % 60 < 10 ? "0" : "") + (s % 60);
    }
}
