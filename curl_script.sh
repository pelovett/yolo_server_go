curl -X POST http://localhost:8080/upload \
	-F "file=@/home/peter/Projects/yolo_server_go/frog.jpg" \
	-H "Content-Type: multipart/form-data"
