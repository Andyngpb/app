#------------------------------------------------------------------------------
# To run your web server, open up your terminal / command prompt
# and type:
#    cd <path to this file>
#    python main.py
#
#------------------------------------------------------------------------------

from flask import Flask, flash, request, redirect, url_for, Response
import requests
import os
import json

# Configure our application 
#
spec_server_url = "http://172.17.0.2:8501/"
mfcc_server_url = "http://172.17.0.3:8503/"

# Initialize our Flask app.
# NOTE: Flask is used to host our app on a web server, so that
# we can call its functions over HTTP/HTTPS.
#
app = Flask(__name__)

#------------------------------------------------------------------------------
# This is our predict URL.
#
# This is just a pass-through (in other words a reverse proxy) that routes
# the entire JSON body to our tensorflow/serving container, retrieves the
# predictions and sends the predictions as-is to the caller.
#------------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    spec_model_wgt = 0.7
    mfcc_model_wgt = 1-spec_model_wgt

    # In this deployment, we are not using file uploads.
    # Instead we will extract JSON directly from the body,
    # which makes things a little easier.
    #
    x = request.get_json()
  
    # TODO:
    # Set the IP address of the Docker container with the tensorflow/serving
    # image to connect to.
    #
    url_spec = spec_server_url + 'v1/models/model/versions/1:predict'
    url_mfcc = mfcc_server_url + 'v1/models/model/versions/1:predict'

    # Set the headers
    #
    headers = {'Content-Type': 'application/json'}

    # POST our JSON coming from the client application
    # to the tensorflow/serving container.
    #
    response_spec = requests.post(url = url_spec, headers = headers, json = x);
    response_mfcc = requests.post(url = url_mfcc, headers = headers, json = x);

    # If successful
    if (response_spec.status_code == 200) && (response_mfcc.status_code == 200) :

        # Retrieve the response and send it back as-is to
        # the calling application.
        #
        yhat_spec = spec_model_wgt * response_spec
        yhat_mfcc = mfcc_model_wgt * response_mfcc
        yhat = np.sum([yhat_spec, yhat_mfcc], axis=0)
        return Response(json.dumps(yhat), mimetype='application/json')

    return Response("{}", mimetype='application/json')                           



#------------------------------------------------------------------------------
# This starts our web server.
# Although we are running this on our local machine,
# this can technically be hosted on any VM server in the cloud!
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=80)


