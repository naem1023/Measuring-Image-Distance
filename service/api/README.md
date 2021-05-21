# Image api server

### Upload and register task to worker
```
$ curl 127.0.0.1:8080/files -F "files=@./README.md"

# Response:
# { files: [ {'original': 'README.md', 'stored': 'XXXX.md'} ] }
```

### Get result of processed by worker
```
$ curl 127.0.0.1:8080/files/XXXX.md/:result

# Response: (Result file uploaded by worker)
# XXXXXX
```