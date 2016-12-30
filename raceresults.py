"""How to use this script
(1) Download race result files by RaceResults.download()
(2) Manually extract text files from lzh files
(3) Move the text files to ./data directory
(4) RaceResults.load() will parse the text files
"""

import numpy as np
import pandas as pd
import urllib
import os
import time
import glob
import collections
# import patoolib

class RaceResults:
    def __init__(self):
        self.baseuri = "http://www1.mbrace.or.jp/od2/K/%s/k%s.lzh" # http://www1.mbrace.or.jp/od2/K/201612/k161201.lzh
        self.results = [] # List of (Racers, 1-2-3)
        self.id2index = None

    def download(self, start, end):
        period = pd.date_range(start, end)

        for date in period:
            # Get file from the website
            dirname = date.strftime("%Y%m")
            lzhname = date.strftime("%y%m%d")
            uri = self.baseuri % (dirname, lzhname)
            savename = "./data/results/lzh/%s.lzh" % lzhname
            if not os.path.exists(savename):
                print("Send request to", uri)
                urllib.request.urlretrieve(uri, savename)
                time.sleep(3)

            # The following unpack part didn't work my Windows environment...
            # Unpack lzh files
            # unpackedname = "./data/results/K%s.TXT" % lzhname
            # if not os.path.exists(unpackedname):
            #     print("Unpacking", savename)
            #     patoolib.extract_archive(savename, outdir="./data/results")

    def load(self):
        collection = []
        for filename in glob.glob("./data/results/K16*.TXT"):
            with open(filename, "r", encoding="shift_jis") as f:
                remaining = -1
                oddscount = -1
                for line in f:
                    if line.startswith("----"):
                        remaining = 6
                        oddscount = 9
                        positions = [None] * 6
                        top3 = [None] * 3
                        odds = []
                    elif remaining > 0:
                        elems = line.replace("\u3000", "").split()
                        id = int(elems[2])
                        pos = int(elems[1]) - 1
                        positions[pos] = id
                        if elems[0] == "01": top3[0] = pos
                        elif elems[0] == "02": top3[1] = pos
                        elif elems[0] == "03": top3[2] = pos
                        collection.append(id)
                        remaining -= 1
                    elif oddscount > 0:
                        elems = line.split()
                        if len(elems) > 0:
                            try:
                                if oddscount == 8:
                                    odds.append((elems[1], int(elems[2]))) # 複勝1
                                    odds.append((elems[3], int(elems[4]))) # 複勝2
                                elif oddscount == 4 or oddscount == 3:
                                    odds.append((elems[0], int(elems[1]))) # 拡連複2, 3
                                else:
                                    # 単勝, 2連単, 2連複, 拡連複1, 3連単, 3連複
                                    odds.append((elems[1], int(elems[2])))
                                oddscount -= 1
                            except:
                                oddscount = -1 # ignore this
                    elif remaining == 0 and oddscount == 0:
                        valid = (len(odds) == 10)
                        for check in positions + top3:
                            if check is None:
                                valid = False
                                break
                        if valid:
                            self.results.append((positions, top3, odds))
                        remaining = -1
                        oddscount = -1

        race_count = collections.Counter(collection)
        race_count[10000] = 0

        remove = []
        for k, v in race_count.items():
            if v < 10:
                remove.append(k)
                race_count[10000] += 0 # Merge with unknown racer (=No 10000)

        for k in remove:
            race_count.pop(k)

        i = 0
        for k in race_count.keys():
            race_count[k] = i
            i += 1

        self.id2index = race_count

    def get_input_length(self):
        return len(self.id2index)

    def get_input(self, id):
        return self.id2index.get(id, self.id2index[10000])

if __name__ == "__main__":
    r = RaceResults()
    # r.download("2016-01-01", "2016-12-27")
    r.load()
