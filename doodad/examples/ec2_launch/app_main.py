import os

import secretlib

import doodad as dd

print("Launching app_main!")

# These are arguments passed in from launch_python
args_dict = dd.get_args()
print("My args are:", args_dict)

# Test proper mounting
out_dir = args_dict["output_dir"]
print(
    "Writing secret (%s) to output dir (%s)"
    % (secretlib.SECRET, os.path.realpath(out_dir))
)
with open(os.path.join(out_dir, "my_secret.txt"), "w") as f:
    f.write(secretlib.SECRET)
print("Done!")
