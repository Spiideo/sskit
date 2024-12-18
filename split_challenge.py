import json

with open("data/SoccerNet/SpiideoSynLoc/annotations/challenge_private.json") as fd:
    data = json.load(fd)
del data['annotations']
with open("data/SoccerNet/SpiideoSynLoc/annotations/challenge_public.json", "w") as fd:
    json.dump(data, fd)