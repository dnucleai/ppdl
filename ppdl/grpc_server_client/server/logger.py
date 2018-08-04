class Logger:

    def _log(self, level, s):
        print("[{}] {}".format(level, s))

    def error(self, s):
        self._log("ERROR", s)

    def warn(self, s):
        self._log("warn", s)

    def info(self, s):
        self._log("info", s)

    def debug(self, s):
        self._log("debug", s)


singleton = Logger()
