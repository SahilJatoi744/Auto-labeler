# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""Dataset management API routes."""

import json
import shutil
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import DatasetInfo
from ..services.preprocessor import get_preprocessor
from ..services.platform import get_platform_service

logger = get_logger("api.datasets")
router = APIRouter(tags=["datasets"])

preprocessor = get_preprocessor()
platform = get_platform_service()


@router.post("/datasets/upload", response_model=DatasetInfo)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None)
):
    """
    Upload and validate a dataset (as ZIP file).

    - **file**: ZIP file containing images
    - **dataset_name**: Optional name for the dataset
    """
    try:
        upload_id = str(uuid4())[:12]
        upload_dir = settings.UPLOAD_DIR / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        zip_path = upload_dir / "upload.zip"
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        import zipfile
        extract_dir = upload_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        image_dir = extract_dir
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            image_dir = subdirs[0]

        dataset_info = preprocessor.process_dataset(
            image_dir,
            dataset_name=dataset_name or file.filename.replace('.zip', ''),
            dataset_id=upload_id,
            max_workers=settings.MAX_WORKERS
        )

        dataset_info.id = upload_id
        zip_path.unlink()

        logger.info(f"Dataset uploaded: {dataset_info.id}")
        version = platform.create_dataset_version(
            dataset_info.id,
            project_id,
            "v1",
            "upload",
            {
                "name": dataset_info.name,
                "total_images": dataset_info.total_images,
                "valid_images": dataset_info.valid_images,
                "formats": dataset_info.formats,
            },
        )
        platform.record_lineage(
            dataset_info.id,
            version["id"],
            "dataset.upload",
            {"filename": file.filename},
            {"valid_images": dataset_info.valid_images, "corrupted_images": dataset_info.corrupted_images},
        )
        platform.record_metric("datasets.uploaded", 1, {"dataset_id": dataset_info.id})
        return dataset_info

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/upload-folder", response_model=DatasetInfo)
async def upload_dataset_folder(
    files: List[UploadFile] = File(...),
    dataset_name: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None)
):
    """Upload multiple images directly (without ZIP)."""
    try:
        upload_id = str(uuid4())[:12]
        upload_dir = settings.UPLOAD_DIR / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        dataset_info = preprocessor.process_dataset(
            upload_dir,
            dataset_name=dataset_name or f"Dataset_{upload_id}",
            dataset_id=upload_id,
            max_workers=settings.MAX_WORKERS
        )

        dataset_info.id = upload_id
        version = platform.create_dataset_version(
            dataset_info.id,
            project_id,
            "v1",
            "folder_upload",
            {"name": dataset_info.name, "total_images": dataset_info.total_images, "valid_images": dataset_info.valid_images},
        )
        platform.record_lineage(
            dataset_info.id,
            version["id"],
            "dataset.folder_upload",
            {"files": len(files)},
            {"valid_images": dataset_info.valid_images, "corrupted_images": dataset_info.corrupted_images},
        )
        platform.record_metric("datasets.uploaded", 1, {"dataset_id": dataset_info.id})
        logger.info(f"Dataset uploaded (folder): {dataset_info.id}")
        return dataset_info

    except Exception as e:
        logger.error(f"Folder upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List all processed datasets."""
    datasets = []

    if not settings.UPLOAD_DIR.exists():
        return []

    for metadata_file in settings.UPLOAD_DIR.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                datasets.append(DatasetInfo(
                    id=data["dataset_id"],
                    name=data["name"],
                    path=data["path"],
                    total_images=data["total_images"],
                    valid_images=data["valid_images"],
                    corrupted_images=data["corrupted_images"],
                    total_size_mb=round(sum(img["size_bytes"] for img in data["images"]) / (1024 * 1024), 2) if "images" in data else 0,
                    formats={},
                    status="ready"
                ))
        except Exception as e:
            logger.error(f"Error reading metadata file {metadata_file}: {e}")

    return datasets


@router.get("/datasets/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str):
    """Get dataset information."""
    metadata_path = settings.UPLOAD_DIR / f"{dataset_id}_metadata.json"

    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    with open(metadata_path) as f:
        data = json.load(f)

    return DatasetInfo(
        id=data["dataset_id"],
        name=data["name"],
        path=data["path"],
        total_images=data["total_images"],
        valid_images=data["valid_images"],
        corrupted_images=data["corrupted_images"],
        status="ready"
    )


@router.get("/datasets/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str):
    """Get detailed statistics for a dataset."""
    dataset_info = await get_dataset(dataset_id)
    stats = preprocessor.get_image_statistics(dataset_info)
    return stats


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset and all associated data."""
    try:
        metadata_path = settings.UPLOAD_DIR / f"{dataset_id}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()

        upload_dir = settings.UPLOAD_DIR / dataset_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

        return {"message": f"Dataset {dataset_id} deleted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
