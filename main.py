#!/bin/python
from flask import Flask, jsonify, request
from flask_caching import Cache
from waitress import serve
from wtforms import Form, validators, StringField
from app import read_number_plates

app = Flask(__name__)
app.config.from_mapping(
    {
        "DEBUG": True,  # some Flask specific configs
        "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
        "CACHE_DEFAULT_TIMEOUT": 300,
    }
)
cache = Cache(app)


@app.route("/status")
def status():
    return "alive"


class ReadForm(Form):
    url = StringField("Url to image", [validators.URL()])


@app.route("/read")
@cache.cached(timeout=86400, query_string=True)
def read():
    form = ReadForm(request.args)

    if form.validate():
        urls = form.url.data
        number_plates, region_names, urls_result = read_number_plates(urls.split(","))

        # print(number_plates, region_names, images_bboxs)

        return jsonify(
            {
                "success": True,
                "url": urls_result,
                "number_plates": number_plates,
                "region_names": region_names,
            }
        )

    return jsonify({"success": False, "errors": form.errors})


def create_app():
    return serve(app)
