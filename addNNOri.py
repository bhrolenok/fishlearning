#addNNOri.py
import os, os.path, string, sys, math
import time

wait_time = 15.0 #in seconds

def dSq(x1,x2,y1,y2):
	return ((float(x1)-float(x2))**2) + ((float(y1)-float(y2))**2)

def rotate(x,y,theta):
	x=float(x)
	y=float(y)
	theta=float(theta)
	xprime = (x*math.cos(theta))-(y*math.sin(theta))
	yprime = (x*math.sin(theta))+(y*math.cos(theta))
	return xprime,yprime

def angle(x,y):
	return math.atan2(float(y),float(x))

def vecTo(fromX,fromY,toX,toY):
	return float(toX)-float(fromX), float(toY)-float(fromY)

def addAllNeighbors(outdir=os.getcwd(),indir=os.getcwd()):
	idf = open(os.path.join(indir,"id.btf")).readlines()
	xpos = open(os.path.join(indir,"xpos.btf")).readlines()
	ypos = open(os.path.join(indir,"ypos.btf")).readlines()
	timage = open(os.path.join(indir,"timage.btf")).readlines()
	timestamp = open(os.path.join(indir,"timestamp.btf")).readlines()
	orifile = open(os.path.join(outdir,"nori.btf"),"w")
	blockStart = 0
	blockEnd = 0
	curLine = 0
	numLines = len(xpos)
	lastTime = time.time()
	lastLine = 0
	while curLine < numLines:
		if curLine >= blockEnd:
			blockStart = blockEnd
			while blockEnd < numLines and timestamp[blockEnd] == timestamp[blockStart]:
				blockEnd += 1
		numNeighbors=0
		for checkLine in xrange(blockStart,blockEnd):
			if idf[curLine] == idf[checkLine]:
				continue
			# tmpVecX,tmpVecY = vecTo(xpos[curLine],ypos[curLine],xpos[checkLine],ypos[checkLine])
			# tmpVecX, tmpVecY = rotate(tmpVecX,tmpVecY,-float(timage[curLine]))
			tmpTheta = float(timage[checkLine]) - float(timage[curLine])
			if numNeighbors > 0:
				orifile.write(" "+str(tmpTheta))
			else:
				orifile.write(str(tmpTheta))
			numNeighbors += 1
		if numNeighbors > 0:
			orifile.write("\n")
		else:
			orifile.write("0\n")
		curLine += 1
		curTime = time.time()
		if (curTime - lastTime) > wait_time:
			print "Line:",curLine,"lps:",float(curLine-lastLine)/(curTime-lastTime), 100*float(curLine)/float(numLines),"%"
			lastLine = curLine
			lastTime = curTime
	orifile.close()

if __name__=="__main__":
	if len(sys.argv) > 4:
		print "Usage: python",sys.argv[0],"[inputDir=os.getcwd()] [outputDir=os.getcwd()]"
	elif len(sys.argv) == 1:
		addAllNeighbors()
	elif len(sys.argv) == 2:
		addAllNeighbors(sys.argv[1])
	else:
		addAllNeighbors(sys.argv[1],sys.argv[2])

