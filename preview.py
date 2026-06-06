import logging
from pathlib import Path

from config import settings
from schemas.models import AssetBundle, AudioAsset, VideoScript, VisualAsset
from pipeline.phase4_editor import VideoEditor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

def main():
    script_file = settings.OUTPUT_DIR / "script.json"
    script = VideoScript.model_validate_json(script_file.read_text())
    
    # Slice just scene 1
    script.scenes = script.scenes[:1]
    
    scene = script.scenes[0]
    
    # Build AssetBundle for scene 1
    audio_asset = AudioAsset(
        scene_id=scene.scene_id,
        path=settings.OUTPUT_DIR / "audio" / f"scene_{scene.scene_id}.wav",
        duration_ms=scene.duration_ms
    )
    
    visual_asset = VisualAsset(
        scene_id=scene.scene_id,
        raw_video_path=settings.OUTPUT_DIR / "video" / f"scene_{scene.scene_id}_veo.mp4",
        anim_video_path=settings.OUTPUT_DIR / "video" / f"scene_{scene.scene_id}_anim.mp4",
    )
    
    bundle = AssetBundle(
        script=script,
        audio_assets=[audio_asset],
        visual_assets=[visual_asset],
        bgm_path=None
    )
    
    editor = VideoEditor()
    editor.run(bundle)

if __name__ == "__main__":
    main()
