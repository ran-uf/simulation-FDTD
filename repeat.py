import subprocess
import time

for j in [34]:
	for i in range(3):
		cmd = "python simulation.py %d %d %d 27" % (i, j, j)
		print(cmd)
		p = subprocess.Popen(cmd)
		p.wait()
		time.sleep(5)
