import subprocess

for i in range(8):
	p = subprocess.Popen("python simulation.py")
	p.wait()
	print(i)