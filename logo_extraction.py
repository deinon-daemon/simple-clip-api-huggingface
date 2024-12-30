import io
import os
import cv2
import uuid
import httpx
import base64
import asyncio
import numpy as np
import urllib.parse
from log import logger
from pydantic import BaseModel
from typing import List, Dict, Tuple
from gcs import get_gcs_record, write_to_gcs
from goo_secrets import secrets
from urllib.parse import urlparse
from PIL import Image, ImageEnhance

DEFAULT_BUCKET = (
    os.environ["BLOODHOUND_BUCKET"]
    if "BLOODHOUND_BUCKET" in os.environ
    else "ravens-db-dev"
)


logo_extensions = [
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".bmp",
    ".tiff",
    ".tif",
]


def is_image_link(href):
    return any(href.lower().endswith(ext) for ext in logo_extensions)


def process_structured_html_to_candidate_logos(html: dict, url: str) -> List[str]:
    img_candidates = []
    img_hash = {}
    explicit_logos = set()
    target_keys = ["ogImgSrcs", "imgSrcs", "candidateLogos"]
    # logger.warning(f"Processing {url}")
    # logger.warning(html)

    def add_new_img_candidates(images: List[str], candidates: List[str]) -> List[str]:
        image_links = [href for href in images if is_image_link(href)]
        image_links = list(
            set(
                [
                    urllib.parse.urljoin(url, href) if href.startswith("/") else href
                    for href in image_links
                    if isinstance(href, str)
                    and ("http" in href or href.startswith("/"))
                ]
            )
        )
        image_links = [img for img in image_links if img not in candidates]

        return image_links

    for t_key in target_keys:
        try:
            if t_key in html and isinstance(html.get(t_key), list):
                img_links = add_new_img_candidates(html.get(t_key), img_candidates)
                img_candidates.extend(list(set(img_links)))
                img_hash[t_key] = list(set(img_links))
        except Exception as e:
            logger.warning(f"Failed to process target_key {t_key}: {e}")
            continue

    # check for explicit logos
    explicit_logos |= set(
        [img for img in img_candidates if "logo" in urlparse(img).path.lower()]
    )

    img_candidates = list(set(img_candidates))
    svg_links = [link for link in img_candidates if link.endswith(".svg")]
    # remove excessive svg links, any svg images past the first couple are going to be used for UI and styling
    if len(svg_links) > 3:
        svg_links = svg_links[:2]
        img_candidates = [
            link
            for link in img_candidates
            if link.endswith(".svg") is False or link in svg_links
        ]

    # filter using html tree logic -- if there are plenty of good candidates tagged as og images, just use those...
    # similarly, if og images &/or img src tags abound, no need to go digging and using compute on other sources for img
    # href tags in the html tree
    enough_images = set()
    for t_key in target_keys:
        try:
            images_of_tag_type = img_hash[t_key]
            enough_images.update(images_of_tag_type)
            if len(images_of_tag_type) > 2:
                break
        except Exception as _e:
            # ignore
            continue

    # final filter
    enough_images = list(enough_images)
    explicit_logos = list(explicit_logos)
    if len(enough_images) > 7:
        enough_images = enough_images[:7]

    if len(explicit_logos) > 4:
        explicit_logos = explicit_logos[:4]
        img_candidates = [img for img in img_candidates if img in explicit_logos]
    else:
        img_candidates = [
            img
            for img in img_candidates
            if img in explicit_logos or img in enough_images
        ]

    # print(img_candidates)
    hostname = urlparse(url).hostname
    default_logo_url = f"https://logo.clearbit.com/{hostname}"
    img_candidates.append(default_logo_url)
    return img_candidates


async def get_image_content(url: str, client: httpx.AsyncClient) -> tuple:
    try:
        response = await client.get(url)
        content_type = response.headers.get("content-type", "").lower()

        if any(img_type in content_type for img_type in ["jpeg", "jpg", "png", "webp"]):
            return content_type, response.content
        else:
            return content_type, None
    except Exception as e:
        logger.warning(f"Error fetching image from {url} using httpx.AsyncClient: {e}")
        return None, None


async def get_screenshot(url: str, client: httpx.AsyncClient) -> dict:
    if "dev" in DEFAULT_BUCKET:
        endpoint = secrets.get("GECKO_URL_DEV")
    else:
        endpoint = secrets.get("GECKO_URL")

    try:
        payload = {
            "url": url,
            "screenshot": True,
            "wait": 7,
        }
        resp = await client.post(endpoint + "/puppeteer", json=payload)

        if resp.status_code > 300:
            logger.warning(f"Error retrieving screenshot for {url} ... {resp}")
            return {"url": url, "status": resp.status_code, "error": True}

        response = resp.json()
        # print(f"puppet response = {response}")
        return {**response, "url": url, "error": False}

    except Exception as e:
        logger.warning(f"httpx.AsyncClient Context Manager Level Error for {url}: {e}")
        return {"url": url, "status": 500, "error": True}


# image processing utils


def load_image_from_bytes(image_bytes):
    """
    Load a PIL Image object from bytes.

    :param image_bytes: Bytes object containing the image data
    :return: PIL Image object
    """
    return Image.open(io.BytesIO(image_bytes))


def is_valid_logo_size(image: Image.Image, min_dimensions=(50, 50)):
    """
    Check if an image meets size requirements before processing.
    Image is valid if either:
    1. Both dimensions meet minimum requirements, or
    2. Total pixel area meets or exceeds the minimum area requirement

    :param image: PIL.Image.Image object containing the image data
    :param min_dimensions: Tuple of (width, height) for minimum acceptable size
    :return: Tuple of (boolean, string) indicating if valid and reason if not
    """
    try:
        # Calculate minimum required area
        min_area = min_dimensions[0] * min_dimensions[1]
        actual_area = image.width * image.height

        # Check if total area meets minimum requirement
        if actual_area >= min_area:
            return True, "Image meets area requirements"

        # If area is too small, check individual dimensions
        if image.width < min_dimensions[0] or image.height < min_dimensions[1]:
            return (
                False,
                f"Image too small: {image.width}x{image.height}. Either meet minimum dimensions {min_dimensions[0]}x{min_dimensions[1]} or total area of {min_area} pixels",
            )

        # Check if image is extremely disproportionate
        aspect_ratio = image.width / image.height
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            return (
                False,
                f"Invalid aspect ratio: {aspect_ratio:.2f}. Should be between 0.2 and 5",
            )

        return True, "Image meets size requirements"

    except Exception as e:
        return False, f"Error analyzing image: {str(e)}"


def process_image(
    image_data, target_size=(250, 250), bg_color=(255, 255, 255), threshold=10
):
    """
    Center the image content, scale to fit within target_size, remove the background,
    and handle low contrast or transparent images.

    :param image_data: Bytes object or string containing the image data
    :param target_size: Tuple of (width, height) for the output image size
    :param bg_color: Background color for padding (default is white)
    :param threshold: Integer value for color difference threshold
    :return: PIL Image object
    """
    # Handle different input types
    if isinstance(image_data, str):
        try:
            # Try base64 first

            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                # If not base64, try other string formats
                if image_data.startswith("b'") or image_data.startswith('b"'):
                    # Remove the b'' or b"" wrapper and evaluate
                    image_data = image_data[2:-1]
                    image_bytes = bytes.fromhex("".join(image_data.split("\\x")[1:]))
                else:
                    # Try direct encoding
                    image_bytes = image_data.encode("utf-8")
        except Exception as e:
            raise ValueError(f"Could not convert string to image bytes: {str(e)}")
    else:
        # Assume it's already bytes
        image_bytes = image_data

    # Load the image from bytes
    image = Image.open(io.BytesIO(image_bytes))

    # Convert image to RGBA if it's not already
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Validate image size first
    is_valid, reason = is_valid_logo_size(image)
    if not is_valid:
        raise ValueError(reason)

    # Convert image to numpy array
    img_array = np.array(image)

    # Check if the image is mostly transparent
    if img_array.shape[2] == 4:
        transparency = img_array[:, :, 3]
        if np.mean(transparency) < 128:
            # Create a white background
            white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            # Paste the original image on the white background
            white_background.paste(image, (0, 0), image)
            image = white_background
            img_array = np.array(image)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    # Perform edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Calculate the percentage of edge pixels
    edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # If there are very few edges, the image might be blank or low contrast
    if edge_percentage < 0.01:
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increase contrast
        img_array = np.array(image)

    # Get the background color (assuming it's the color of the corner pixel)
    bg_pixel = img_array[0, 0]

    # Create a mask where a pixel is True if it's different from the background
    mask = np.abs(img_array[:, :, :3] - bg_pixel[:3]).sum(axis=2) > threshold

    # Find the bounding box of the non-background content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        # Crop the image to the content
        content = image.crop((xmin, ymin, xmax + 1, ymax + 1))
    else:
        content = image

    # Calculate the size to fit the content within the target size
    content_aspect = content.width / content.height
    target_aspect = target_size[0] / target_size[1]

    if content_aspect > target_aspect:
        new_width = target_size[0]
        new_height = int(new_width / content_aspect)
    else:
        new_height = target_size[1]
        new_width = int(new_height * content_aspect)

    # Resize the content
    content_resized = content.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with the target size and background color
    new_image = Image.new("RGBA", target_size, bg_color)

    # Calculate position to paste the resized content
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2

    # Paste the resized content onto the new image
    new_image.paste(content_resized, (paste_x, paste_y), content_resized)

    return new_image


labels = [
    "cat",
    "person",
    "building",
    "a company logo",
    "website",
    "shape",
    "Facebook Logo",
    "Twitter Logo",
]


class CandidateLogo(BaseModel):
    url: str
    storageLink: str
    image: Image.Image
    uid: str | None = None
    scores: List[dict] = [{"label": label, "score": 0.0} for label in labels]

    class Config:
        # PIL.Image.Image lacks a couple dunder methods
        # pydantic depends on to auto generate mappings for class attributes
        # => generic dTypes ... this doesn't concern us because we just care
        # that the type(x) == Image.Image and not something else
        arbitrary_types_allowed = True


class Logo(BaseModel):
    uid: str
    asset_url: str
    source_url: str
    gcs_storage_link: str
    scores: List[dict]


async def collect_image(url: str, client: httpx.AsyncClient) -> CandidateLogo:
    if not url:
        return None
    elif not url.startswith("https:"):
        url = url.replace("http:", "https:")
        if not url.startswith("https:"):
            logger.info(f"Not a valid candidate logo url: {url}")
            return None

    content_type, image_data = await get_image_content(url, client)

    if image_data is not None:
        # Process the image data directly
        scaled_image = process_image(image_data)
        return CandidateLogo(url=url, storageLink=url, image=scaled_image, uid=None)

    # puppeteer screenshot fallback for unusual // hard-to-process content-types (like svg)
    else:
        # print(f"calling puppeteer for screenshot! {url=}")
        screenshot_result = await get_screenshot(url, client)
        link = screenshot_result.get("storageLink")
        _status = screenshot_result.get("status")
        _err_status = screenshot_result.get("error")
        uid = screenshot_result.get("uid")
        if not isinstance(uid, str) or uid == "None":
            return None
        gcs_image_bytes = get_gcs_record(
            uid, DEFAULT_BUCKET, "puppet/screenshots", ".jpg"
        )
        scaled_image = process_image(gcs_image_bytes)
        return CandidateLogo(url=url, storageLink=link, image=scaled_image, uid=uid)


async def collect_images(
    urls: List[str], client: httpx.AsyncClient
) -> List[CandidateLogo]:
    images = []

    tasks = [collect_image(url, client) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for url, r in zip(urls, results):
        if isinstance(r, CandidateLogo):
            images.append(r)
        else:
            logger.info(f"Error processing {url} ... {str(r)}")

    return images


# take collected images and generate "CLIP.py" features from them
async def query_clippy(
    candidate_logo: CandidateLogo, parameters: List[str], client: httpx.AsyncClient
) -> CandidateLogo:
    API_URL = secrets.get("hf_clip_url")
    hf_bearer_key = secrets.get("hf_clip_key")
    if isinstance(hf_bearer_key, str) is False or isinstance(API_URL, str) is False:
        raise ValueError(
            f"URL &/or Bearer auth for CLIP service not set! {API_URL=} || {hf_bearer_key=}"
        )

    headers = {"Authorization": f"Bearer {hf_bearer_key}"}
    # Convert PIL Image to bytes in memory
    image = candidate_logo.image
    with io.BytesIO() as img_byte_arr:
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        # Encode image to base64
        img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

        payload = {"candidate_labels": parameters, "inputs": img_base64}

        try:
            response = await client.post(API_URL, headers=headers, json=payload)
            scores = response.json()
            if "error" in scores or isinstance(scores, list) is False:
                raise ValueError(f"Bad Response from Clippy {scores=}")
            candidate_logo.scores = scores
        except Exception as e:
            logger.info(f"non-fatal clippy error == {e} ... backing off and retrying")
            await asyncio.sleep(5)
            response = await client.post(API_URL, headers=headers, json=payload)
            scores = response.json()
            if "error" in scores:
                raise ValueError(f"Bad Response from Clippy {scores=} || {payload=}")
            candidate_logo.scores = scores

        return candidate_logo


async def calculate_logo(
    candidate_logos: List[str], source_url: str, name: str = ""
) -> CandidateLogo:
    async with httpx.AsyncClient(timeout=45) as client:
        final_results = []
        name_label = ""
        try:
            results = await collect_images(candidate_logos, client)

            if isinstance(results, list) is False or len(results) == 0:
                # logger.warning(
                #     f"NO RESULTS FROM IMAGE COLLECTION => {len(candidate_logos)=}"
                # )
                raise ValueError(f"NO RESULTS FROM IMAGE COLLECTION => {source_url}")

            if isinstance(name, str) and name != "":
                name_label = f"The Logo of {name}"
                clippy_tasks = [
                    query_clippy(r, labels + [name_label], client) for r in results
                ]
            else:
                clippy_tasks = [query_clippy(r, labels, client) for r in results]

            clippy_results = await asyncio.gather(*clippy_tasks, return_exceptions=True)
            for clippy_result in clippy_results:
                try:
                    if isinstance(clippy_result, CandidateLogo) is False:
                        # logger.warning(
                        #     f"Clippy response returned error || {str(clippy_result)}"
                        # )
                        continue

                    final_results.append(clippy_result)
                except Exception as e:
                    logger.warning(f"Error unpacking Clippy response => {e=}")

            try:
                best = sorted(
                    sorted(
                        [
                            r
                            for r in final_results
                            if r.scores[0].get("label")
                            in ["a company logo", name_label]
                        ],
                        key=lambda x: len(x.scores[0].get("label")),
                    ),  # sorts to prioritize name label if != ""
                    key=lambda x: x.scores[0].get("score"),
                    reverse=True,
                )
                # print(best)
                best = best[0]
            except Exception as _e:
                # constraints too strict
                good_features = [name_label, "a company logo", "website"]
                best = sorted(
                    [
                        r
                        for r in final_results
                        if (r.scores[0].get("label") in good_features)
                        and (
                            next(s for s in r.scores if s.get("label") == "person").get(
                                "score"
                            )
                            <= 0.01
                        )
                    ],
                    key=lambda x: x.scores[0].get("score"),
                    reverse=True,
                )[0]

            return best

        except Exception as _e:
            # logger.warning(
            #     f"ERROR FINDING CANDIDATE LOGOS FOR {source_url} || {e} => USING BACKUP"
            # )
            # logger.warning(f"{final_results=}")
            hostname = urlparse(source_url).hostname
            default_logo_url = f"https://logo.clearbit.com/{hostname}"
            default_logo = await collect_image(default_logo_url, client)
            return default_logo


async def get_logo(
    source_url: str, html_doc: dict, name: str = "", return_image: bool = False
) -> Logo | Dict | Tuple[Logo, Image.Image]:
    try:
        candidate_logos = process_structured_html_to_candidate_logos(
            html_doc, source_url
        )
    except Exception as e:
        logger.warning(
            f"error processing html_doc in bullseye_vision.get_logo for {source_url} => {e}"
        )
        return {"url": source_url, "error": str(e)}

    try:
        best_logo = await calculate_logo(candidate_logos, source_url, name)

        # Convert the logo image to bytes
        with io.BytesIO() as img_byte_arr:
            best_logo.image.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()

            # Generate a unique filename
            if isinstance(best_logo.uid, str) is False:
                best_logo.uid = str(uuid.uuid4())

            filename = f"{best_logo.uid}.png"

            # Write to GCS
            options = {"contentType": "image/png"}
            status = write_to_gcs(
                filename,
                img_bytes,
                options,
                dir_name="bullseye_logos",
                bucket_name="eco_one_images",
            )

        if status != 200:
            raise Exception(f"Failed to write logo to GCS. Status: {status}")

        final_logo = Logo(
            uid=best_logo.uid,
            asset_url=source_url,
            source_url=best_logo.url,
            gcs_storage_link=f"https://storage.googleapis.com/eco_one_images/bullseye_logos/{filename}",
            scores=best_logo.scores,
        )
        if return_image is True:
            return final_logo, best_logo.image

        else:
            return final_logo

    except Exception as e:
        # logger.warning(f"Error in bullseye_vision.get_logo for {source_url}: {e}")
        return {"url": source_url, "error": str(e)}
