# gunicorn.conf.py

import os

bind = "0.0.0.0:8947"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 0
loglevel = "info"
accesslog = "-"  
errorlog = "-"   
preload_app = False  
