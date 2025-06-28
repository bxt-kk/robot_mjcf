import json
from bpy import context as C


class DumpPose:

    def __init__(self, fname:str):
        self.fname = fname
        self.active_obj = C.selected_objects[0]

    def __call__(self) -> str:
        pos = tuple(self.active_obj.location)
        quat = tuple(self.active_obj.rotation_euler.to_quaternion())
        text = json.dumps(dict(pos=pos, quat=quat))
        with open(self.fname, 'a') as af:
            af.write(text + '\n')
        return text


if __name__ == "__main__":
    dumper = DumpPose('< filename >')
    dumper()
