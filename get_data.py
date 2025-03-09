"""!@file get_data.py
@breif This file is used to download the data from the google drive link.
@details The file uses the pydrive library to download the data
from the google drive link.
@author Shizhe Xu
@date 28 June 2024
"""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


def authenticate():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("mycreds.txt")
    return GoogleDrive(gauth)


def download_files_in_folder(drive, folder_id, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    file_list = drive.ListFile(
        {"q": f"'{folder_id}' in parents and trashed=false"}
    ).GetList()

    for file in file_list:
        file_path = os.path.join(dest_folder, file["title"])
        if file["mimeType"] == "application/vnd.google-apps.folder":
            print(f'Entering folder: {file["title"]}')
            download_files_in_folder(drive, file["id"], file_path)
        else:
            print(f'Downloading {file["title"]} to {file_path}')
            file.GetContentFile(file_path)


def main():
    drive = authenticate()

    # ID of the root folder with our sharing link
    root_folder_id = "16dPY5ekOK3Zu67Cctm4yY868XlCCqegn"

    # destination folder on your local machine
    local_dest_folder = "downloads"

    download_files_in_folder(drive, root_folder_id, local_dest_folder)
    print("All files and folders downloaded successfully.")


if __name__ == "__main__":
    main()
