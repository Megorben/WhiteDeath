package whitedeath.recorder;

import arc.*;
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
        recorder.register(() ->
            Core.settings.getBool("wd-rec-enabled", true)
            && net.client()
            && (state.rules.pvp || !Core.settings.getBool("wd-rec-pvp-only", true)));
        Events.run(Trigger.update, () -> recorder.keyframeTicks = Core.settings.getInt("wd-rec-keyframe", 60) * 60);
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

        ui.hudGroup.fill(t -> {
            t.bottom().right();
            t.visible(() -> recorder.active() && Core.settings.getBool("wd-rec-indicator", true) && ui.hudfrag.shown);
            t.label(() -> "[scarlet]●[] REC " + formatTime(recorder.seconds())).style(Styles.outlineLabel).margin(6f);
        });
    }

    static String formatTime(float seconds){
        int s = (int)seconds;
        return Strings.format("@:@", s / 60, s % 60 < 10 ? "0" + s % 60 : s % 60);
    }
}
