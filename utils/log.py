#! /usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime, date, timedelta
import logging
import os
from logging.handlers import TimedRotatingFileHandler
import time
import fcntl


class MultiCompatibleTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    重载TimedRotatingFileHandler 类的doRollover方法，可以支持多进程翻转
    """

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        # 兼容多进程并发 LOG_ROTATE
        if not os.path.exists(dfn):
            f = open(self.baseFilename, 'a')
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            if os.path.exists(self.baseFilename) and not os.path.exists(dfn):
                os.rename(self.baseFilename, dfn)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt


class Clogger(logging.Logger):
    FORMAT = '%(message)s'
    SUFFIX = '%Y%m%d_utf8.text'

    def __init__(self, name='ai_error'):
        logging.Logger.__init__(self, name)

    def init(self, log_file):
        self.log_file = log_file
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))

        self.rollover()

    def rollover(self):
        self.handlers = []
        filename = self.log_file + datetime.now().strftime(self.SUFFIX)
        handler = logging.FileHandler(filename)
        format = logging.Formatter(self.FORMAT)
        self.addHandler(handler)
        tomorrow = date.today() + timedelta(days=1)
        self.rolloverAt = datetime.strptime(str(tomorrow), '%Y-%m-%d')

    def record(self, data):
        t = int(time.time())
        dt = datetime.fromtimestamp(t)
        dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        msg = '[{dt_str}] {data}'.format(dt_str=dt_str, data=data)
        if datetime.now() > self.rolloverAt:
            self.rollover()
        logging.Logger.info(self, msg)

