build:
	docker build -t faceshape:v1 --force-rm -f Dockerfile .

run:
	docker run -d --name faceshape -p 8000:8000 faceshape:v1 
