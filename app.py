import os
import sys
import tempfile
from urllib.request import urlopen

# NomeroffNet path
nomeroff_net_dir = os.path.abspath('../nomeroff-net')
sys.path.append(nomeroff_net_dir)
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

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

def read_number_plates(url):
    global number_plate_detection_and_reading

    with tempfile.NamedTemporaryFile() as fp:
        with urlopen(url) as response:
            fp.write(response.read())

        result = number_plate_detection_and_reading([fp.name])

    (images, images_bboxs,
       images_points, images_zones, region_ids,
       region_names, count_lines,
       confidences, texts) = unzip(result)

    return texts[0], region_names[0], images_bboxs[0]
