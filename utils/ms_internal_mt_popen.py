import subprocess

filename = "ms_internal_mt.py"
while True:
	p = subprocess.Popen('python '+filename, shell=True).wait()
	if p!=0:
		print("An error occured in the translation process. Restaring....")
		continue
	else:
		print("The translation process is Done")
		break