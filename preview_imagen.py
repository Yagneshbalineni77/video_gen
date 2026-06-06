import logging
from pathlib import Path

from config import settings
from pipeline.phase3_assets import AssetFactory
from pipeline.phase4_editor import VideoEditor
from schemas.models import VideoScript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

def main():
    script_file = settings.OUTPUT_DIR / "script.json"
    script = VideoScript.model_validate_json(script_file.read_text())
    
    # Run only for scene 1
    script.scenes = script.scenes[:1]
    
    # Run Phase 3 (will hit Imagen3 then Veo3)
    factory = AssetFactory()
    bundle = factory.run(script)
    
    # Run Phase 4
    editor = VideoEditor()
    editor.run(bundle)

if __name__ == "__main__":
    main()
