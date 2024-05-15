import os
import sys
import tempfile
from urllib.request import urlopen

# NomeroffNet path
nomeroff_net_dir = os.path.abspath('../nomeroff-net')
sys.path.append(nomeroff_net_dir)
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
import itertools

import logging
logger = logging.getLogger('waitress')

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv",
                 presets={
                        "eu_ua_2004_2015_efficientnet_b2": {
                            "for_regions": ["eu_ua_2015", "eu_ua_2004"],
                            "model_path": "latest"
                         },
                        "eu_ua_1995_efficientnet_b2": {
                            "for_regions": ["eu_ua_1995"],
                            "model_path": "latest"
                        },
                        "eu_ua_custom_efficientnet_b2": {
                            "for_regions": ["eu_ua_custom"],
                            "model_path": "latest"
                        },
                        "xx_transit_efficientnet_b2": {
                            "for_regions": ["xx_transit"],
                            "model_path": "latest"
                        },
                        "eu_efficientnet_b2": {
                            "for_regions": ["eu", "xx_transit", "xx_unknown", "md", "am", "by"],
                            "model_path": "latest"
                        },
                 },
                 one_preprocess_for_ocr_and_classification=False)

def read_number_plates(urls):
    global number_plate_detection_and_reading

    files = []
    for url in urls:
        # with tempfile.NamedTemporaryFile() as fp:
        fp = tempfile.NamedTemporaryFile()
        try:
            with urlopen(url) as response:
                fp.write(response.read())
            files.append(fp)
        except Exception as e:
            logger.error(e)
    # logger.info(files)
    if len(files) == 0:
        return [], []
    results = number_plate_detection_and_reading([fp.name for fp in files])
    for file in files:
        file.close()

    (images, images_bboxs,
       images_points, images_zones, region_ids,
       region_names, count_lines,
       confidences, texts) = unzip(results)

    numberplates = {}

    numberplates_presence = []

    for url_id in range(len(urls)):
        if len(texts[url_id]) > 0:
            numberplates_presence.append(urls[url_id])

    logger.info(texts)
    # logger.info(images_bboxs)
    for idx, images_bbox in enumerate(images_bboxs):
        # logger.info('========')
        areas = [abs(x2 - x1) * abs(y2 - y1) for x1, y1, x2, y2, confidence, class_id in images_bbox]
        # logger.info(texts[idx])
        # logger.info(areas)
        if len(areas) > 0:
            # logger.info("-----")
            max_numberplate_area_id = areas.index(max(areas))
            try:
                max_numberplate = texts[idx][max_numberplate_area_id]
                if max_numberplate in numberplates:
                    numberplates[max_numberplate] += 1
                else:
                    numberplates[max_numberplate] = 1
            except IndexError:
                continue

    logger.info(numberplates)

    if len(numberplates.keys()) > 0:
        max_numberplate = max(numberplates, key=numberplates.get)
    else:
        max_numberplate = None

    logger.info(max_numberplate)

    final_result = []

    if max_numberplate is None:
        final_result = []
    else:
        for text, count in numberplates.items():
            if count == numberplates[max_numberplate]:
                final_result.append(text)

    logger.info(final_result)

    return final_result, list(itertools.chain(*region_names)), numberplates_presence
