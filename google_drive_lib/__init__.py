from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from .generate_token import get_token


def get_folder_id(file_name: str, servicee, parent_folder_id: str = "") -> str:
    response = (
        servicee.files()
        .list(
            q=f"name='{file_name}' and mimeType='application/vnd.google-apps.folder'",
            spaces="drive",
        )
        .execute()
    )

    if not response["files"]:
        file_metadata = {
            "name": file_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id] if parent_folder_id else "",
        }
        file = servicee.files().create(body=file_metadata, fields="id").execute()
        return (file.get("id"), False)

    return (response["files"][0]["id"], True)


def save_results_files(
    scenario_dir_names: list, servicee, folder_name: str, folder_ids: list
) -> None:
    for scenario_dir in scenario_dir_names:
        scenario_dir_name = str(scenario_dir).rsplit("/", maxsplit=1)[-1]
        folder_ids[scenario_dir_name], _ = get_folder_id(
            scenario_dir_name, servicee, folder_ids[folder_name]
        )

        for file_name in Path(f"./{scenario_dir}").iterdir():
            file_name_str = str(file_name).rsplit("/", maxsplit=1)[-1]
            cur_id, exists = get_folder_id(
                file_name_str.replace(".csv", ""),
                servicee,
                folder_ids[scenario_dir_name],
            )
            if exists:
                continue
            file_metadata = {"name": file_name_str, "parents": [cur_id]}
            media = MediaFileUpload(file_name)
            servicee.files().create(
                body=file_metadata, media_body=media, fields="id"
            ).execute()
            print(f"Backed up file: {file_name}")


def save_metrics_files(scenario_dir_names: list, servicee, folder_ids: list) -> None:
    for scenario_dir in scenario_dir_names:
        scenario_dir_name = str(scenario_dir).rsplit("/", maxsplit=1)[-1]

        for file_name in Path(f"./{scenario_dir}").iterdir():
            file_name_str = str(file_name).rsplit("/", maxsplit=1)[-1]
            folder_ids[file_name_str], _ = get_folder_id(
                file_name_str, servicee, folder_ids[scenario_dir_name]
            )

            files = (
                servicee.files()
                .list(
                    q="mimeType='image/png' or mimeType='application/pdf'",
                    spaces="drive",
                    fields="files(id, name)",
                )
                .execute()
            )

            gdrive_file_names = [file.get("name") for file in files.get("files", [])]

            for img_path in Path(file_name).iterdir():
                img = f"{str(img_path).rsplit('/', maxsplit=1)[-1].replace('.pdf','')}_for_{file_name_str}.pdf"
                if img in gdrive_file_names:
                    continue
                file_metadata = {"name": img, "parents": [folder_ids[file_name_str]]}
                media = MediaFileUpload(img_path)
                servicee.files().create(
                    body=file_metadata, media_body=media, fields="id"
                ).execute()
                print(f"Backed up file: {img_path}")


def upload_to_drive():
    creds = get_token()

    try:
        service = build(serviceName="drive", version="v3", credentials=creds)
        folder_idss = {}
        folder_idss["top_parent"], _ = get_folder_id("AsyncFLFolder2022", service)
        folder_idss["results"], _ = get_folder_id(
            "results", service, folder_idss["top_parent"]
        )
        save_results_files(
            list(Path("./results").iterdir()), service, "results", folder_idss
        )
        save_metrics_files(
            list(Path("./results_metrics").iterdir()), service, folder_idss
        )

    except HttpError as e:
        print(f"Error: {e}")
