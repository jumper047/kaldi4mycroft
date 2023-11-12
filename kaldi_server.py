#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#
# Copyright 2017 Guenter Bartsch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#
#
# simple speech recognition http api server - designed
# to be drop-in replacement  for kaldi-gstreamer-server
#
# WARNING:
#     right now, this supports a single client only - needs a lot more work
#     to become (at least somewhat) scalable
#
#
# Returns:
#
# * 400 if request is invalid
# * 200 OK
# * 201 OK {"hstr": "hello world", "confidence": 0.02,\
# "audiofn": "data/recordings/anonymous-20170105-rec/wav/de5-005.wav"}
#
# Example:
#
#  curl -X POST --data-binary @2830-3980-0043.wav http://localhost:8301/decode
#

import logging
import json
import wave
import struct

from time import time
from optparse import OptionParser
from setproctitle import setproctitle
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from StringIO import StringIO

from kaldiasr.nnet3 import KaldiNNet3OnlineModel, KaldiNNet3OnlineDecoder
import numpy as np

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 8301

DEFAULT_MODEL_DIR = '/opt/kaldi_res/kaldi-generic-en-tdnn_250-r20190609'
DEFAULT_MODEL = 'model'

SAMPLE_RATE = 16000

PROC_TITLE = 'kaldi_server'

#
# globals
#

decoder = None  # kaldi nnet3 online decoder
mycroft_output = False  # which output format to use

class SpeechHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_error(400, 'Invalid request')

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):

        global decoder

        logging.debug("POST %s" % self.path)

        if self.path == "/decode":

            data = StringIO(self.rfile.read(
                int(self.headers.getheader('content-length'))))
            wavefile = wave.open(data, 'rb')
            nframes = wavefile.getnframes()
            samples = struct.unpack_from('<%dh' % nframes,
                                         wavefile.readframes(nframes))

            # print data

            hstr = ''
            confidence = 0.0

            decoder.decode(SAMPLE_RATE, np.array(samples, dtype=np.float32),
                           True)

            hstr, confidence = decoder.get_decoded_string()

            logging.debug("** %9.5f %s" % (confidence, hstr))

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            reply = {"status": 0,
                     "hypotheses": [{"utterance": hstr}],
                     "id": "000000"}
            self.wfile.write(json.dumps(reply))
            return


if __name__ == '__main__':

    setproctitle(PROC_TITLE)

    #
    # commandline
    #

    parser = OptionParser("usage: %prog [options] ")

    parser.add_option("-v",
                      "--verbose",
                      action="store_true",
                      dest="verbose",
                      help="verbose output")

    parser.add_option("-H",
                      "--host",
                      dest="host",
                      type="string",
                      default=DEFAULT_HOST,
                      help="host, default: %s" % DEFAULT_HOST)

    parser.add_option("-p",
                      "--port",
                      dest="port",
                      type="int",
                      default=DEFAULT_PORT,
                      help="port, default: %d" % DEFAULT_PORT)

    parser.add_option("-d",
                      "--model-dir",
                      dest="model_dir",
                      type="string",
                      default=DEFAULT_MODEL_DIR,
                      help="kaldi model directory, default: %s" %
                      DEFAULT_MODEL_DIR)

    parser.add_option("-m",
                      "--model",
                      dest="model",
                      type="string",
                      default=DEFAULT_MODEL,
                      help="kaldi model, default: %s" % DEFAULT_MODEL)

    (options, args) = parser.parse_args()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    kaldi_model_dir = options.model_dir
    kaldi_model = options.model

    #
    # setup kaldi decoder
    #

    start_time = time()
    logging.info('%s loading model from %s ...' %
                 (kaldi_model, kaldi_model_dir))
    nnet3_model = KaldiNNet3OnlineModel(kaldi_model_dir, kaldi_model)
    logging.info('%s loading model... done. took %fs.' %
                 (kaldi_model, time() - start_time))
    decoder = KaldiNNet3OnlineDecoder(nnet3_model)

    #
    # run HTTP server
    #

    try:
        server = HTTPServer((options.host, options.port), SpeechHandler)
        logging.info('listening for HTTP requests on %s:%d' %
                     (options.host, options.port))

        # wait forever for incoming http requests
        server.serve_forever()

    except KeyboardInterrupt:
        logging.error('^C received, shutting down the web server')
        server.socket.close()
