#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

""" This utility facilitates downloading and installing SUMO Challenge dataset.

It downloads all zip files in parallel, then unzips files in parallel, cleaning
up after each successful unzipping to save space.

Usage: python3 download_sumo_datasets.py <destination_dir> <sumo_server_name>

Example:
    python3 download_sumo_datasets.py ~/download https://example.com/

Note: You will receive the server name after requesting access to the data through
      the sign-up process.

IMPORTANT: Your destination folder must have enough space to hold all the entire
           dataset (Roughly 2.4 TB)

Note: This python script depends on Python 3.3+  and 'requests' package. If your
      local python3 does not have 'requests', you can easily install it using:
      'pip3 install requests'. It is not tested with python 2.

"""

import argparse
import glob
import logging
import os
import shutil
import zipfile
from functools import partial
from multiprocessing import Pool

import requests


# config file on the server
CONFIG_FILE_PATH = "config/sumochallenge.json"

# chunk size when reading zip files from the server
CHUNK_SIZE = 1024 * 1024 * 500  # 500 MB

# maximum number of cores to use for parallel processing
MAX_CORES = 3

# maximum number of re-tries for download and unzip methods before
# giving up
MAX_RETRIES = 3

LOGFORMAT = "%(asctime)-15s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFORMAT)


def remove_existing_files(zip_files, existing_zip_files):
    """Removes entries in <zip_files> that exist in <existing_zip_files>."""
    return [zip_file for zip_file in zip_files if zip_file not in existing_zip_files]


def read_config(args):
    """Reads the configuration file from the server, which contains the list of
    files to download."""
    response = requests.get(args.server_name + CONFIG_FILE_PATH)
    if response.status_code != 200:
        raise RuntimeError(
            "Cannot read config file from the server. Check your connection "
            "or proxy settings (if needed)"
        )
    config = response.json()

    if config["version"] != "v1":
        raise RuntimeError(
            "A new version of config file is deployed. "
            "Please download the most recent script from "
            "https://github.com/facebookresearch/sumo-challenge"
        )
    args.version = config["version"]
    args.training_input = config["training_input"]
    args.training_ground_truth = config["training_ground_truth"]
    args.test_input = config["test_input"]


def download_one(args, zip_filename):
    """Downloads one zip file in a single process."""
    try:
        logging.info("Downloading {}.".format(zip_filename))
        server_url = "{}{}/{}".format(args.server_name, args.version, zip_filename)
        response = requests.get(server_url, stream=True)
        if not response.ok:
            raise RuntimeError(
                "Cannot read zip file {} from the server".format(zip_filename)
            )
        total_length = int(response.headers.get("content-length"))
        with open(os.path.join(args.destination_dir, zip_filename), "wb") as f:
            if total_length == 0:
                f.write(response.content)
            else:
                for idx, chunk in enumerate(
                        response.iter_content(chunk_size=CHUNK_SIZE)
                ):
                    logging.debug(
                        " -- {} {:.0%}".format(
                            zip_filename, (idx + 1) * CHUNK_SIZE / total_length
                        )
                    )
                    if chunk:
                        f.write(chunk)
                        f.flush()

    except KeyboardInterrupt:
        if os.path.exists(os.path.join(args.destination_dir, zip_filename)):
            os.remove(os.path.join(args.destination_dir, zip_filename))
        raise

    except Exception as e:
        logging.error(e)
        if os.path.exists(os.path.join(args.destination_dir, zip_filename)):
            os.remove(os.path.join(args.destination_dir, zip_filename))


def unzip_one(args, file):
    """Unzips one file in a single process."""
    try:
        zip_file_path = os.path.join(args.destination_dir, file)
        logging.info("Extracting {}.".format(zip_file_path))
        with zipfile.ZipFile(zip_file_path, "r") as zip:
            zip.extractall(path=args.destination_dir)
        os.remove(zip_file_path)
        with open(
            os.path.join(
                args.destination_dir, ".cache", file.replace(".zip", ".cache")
            ),
            "w",
        ) as f:
            f.write("complete")

    except KeyboardInterrupt:
        raise

    except Exception as e:
        logging.error(e)


def download_zips_in_parallel(args):
    """Uses a pool of processes to download zip files from the server in parallel."""
    try:
        # grab the list of remining zips to be downloaded
        _, _, remaining_zips, _ = get_zip_file_count(args)
        if len(remaining_zips) == 0:
            logging.info("Download complete.")
            return

        retries = 0
        while retries < MAX_RETRIES:
            retries += 1
            # create a pool of 3 processes
            pool = Pool(MAX_CORES)
            logging.info(
                "Attempt #{}: downloading {} zip files in parallel".format(
                    retries, len(remaining_zips)
                )
            )
            pool.map_async(partial(download_one, args), remaining_zips)
            pool.close()
            pool.join()

            # verify that all zips are downloaded
            _, _, remaining_zips, _ = get_zip_file_count(args)
            if len(remaining_zips) == 0:
                logging.info("Download complete.")
                return

        # final check
        _, _, remaining_zips, _ = get_zip_file_count(args)
        if len(remaining_zips) > 0:
            logging.error(
                "Unable to download all zip files after {} tries. "
                "(remaining: {})".format(MAX_RETRIES, len(remaining_zips))
            )
            return

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise

    except Exception as e:
        logging.error(e)
        pool.terminate()
        pool.join()


def unzip_files_in_parallel(args):
    """Uses a pool of processes to unzip downloaded files in parallel."""
    try:
        # grab the list of existing zips
        _, existing_zip, _, _ = get_zip_file_count(args)
        if len(existing_zip) == 0:
            logging.info("Extraction complete.")
            return

        retries = 0
        while retries < MAX_RETRIES:
            retries += 1
            # create a pool of 3 processes
            pool = Pool(MAX_CORES)
            logging.info(
                "Attempt #{}: extracting {} zip files in parallel".format(
                    retries, len(existing_zip)
                )
            )
            pool.map_async(partial(unzip_one, args), existing_zip)

            pool.close()
            pool.join()
            # check to see if all files are processed
            _, existing_zip, _, _ = get_zip_file_count(args)
            if len(existing_zip) == 0:
                logging.info("Extraction complete.")
                return

        # final verification
        all_zips, _, _, extracted_zips = get_zip_file_count(args)
        if len(all_zips) > len(extracted_zips):
            logging.error(
                "Unable to extract all zip files after {} tries. "
                "(remaining: {})".format(
                    MAX_RETRIES, len(all_zips) - len(extracted_zips)
                )
            )
            return

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise

    except Exception as e:
        logging.error(e)
        pool.terminate()
        pool.join()


def get_zip_file_count(args):
    """Returns a tuple of four lists: all zip files, already downloaded zip files,
    remaining zip files to be downloaded, and extracted zip files."""
    all_zip_files = args.training_input + args.training_ground_truth + args.test_input
    existing_zip_files = [
        os.path.basename(x)
        for x in glob.glob(os.path.join(args.destination_dir, "*.zip"))
    ]
    extracted_zip_files = [
        os.path.basename(x.replace(".cache", ".zip"))
        for x in glob.glob(os.path.join(args.cache_folder, "*.cache"))
    ]
    remaining_zip_files_to_be_downloaded = remove_existing_files(
        all_zip_files, existing_zip_files + extracted_zip_files
    )
    return (
        all_zip_files,
        existing_zip_files,
        remaining_zip_files_to_be_downloaded,
        extracted_zip_files,
    )


def create_target_folder(args):
    """Creates target folder and cache folder if necessary."""
    if not os.path.isdir(args.destination_dir):
        os.makedirs(args.destination_dir)

    args.cache_folder = os.path.join(args.destination_dir, ".cache")
    if not os.path.isdir(args.cache_folder):
        os.mkdir(args.cache_folder)


def calculate_disk_space_needed(args):
    """Calculates the remaining disk space needed to proceed and stops if there
    is not enough disk space for the operation."""

    _, _, remaining_zips, _ = get_zip_file_count(args)

    usage = shutil.disk_usage(args.destination_dir)
    free = usage[2] / 1024 / 1024 / 1024  # in GB
    # on average 6.5G per zip file + 10 GB for extraction
    required_space = len(remaining_zips) * 6 + 10
    if free < required_space:
        raise RuntimeError(
            "Your destination folder needs to have at least {:,}GB of free space.".format(
                required_space
            )
        )


def get_args():
    """Creates an ArgumentParser and returns the arguments namespace."""
    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument("destination_dir", help="Destination folder for SUMO data.")
    parser.add_argument("server_name", help="The url of sumochallenge server.")
    return parser.parse_args()


if __name__ == "__main__":

    try:
        args = get_args()
        read_config(args)
        create_target_folder(args)
        calculate_disk_space_needed(args)
        download_zips_in_parallel(args)
        unzip_files_in_parallel(args)

    except KeyboardInterrupt:
        logging.warning("Download interrupted by user.")

    except Exception as e:
        logging.error("Program terminated with the following error: {}".format(e))
