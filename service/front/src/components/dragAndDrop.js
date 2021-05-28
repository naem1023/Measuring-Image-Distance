import React, {useCallback, useState, useEffect} from 'react'
import axios from 'axios'
import {useDropzone} from 'react-dropzone'

function MyDropzone({setResultPath: setPropResultPath}) {

  const [uploadedImage, setUploadedImage] = useState("");
  const [processedStatus, setProcessedStatus] = useState(0);
  const [intervalId, setIntervalId] = useState(0);
  const [resultFilePath, setResultFilePath] = useState("");

  const onDrop = useCallback(async acceptedFiles => {
    // Do something with the files
    const formData = new FormData();
    const config = {
      header: {
        "content-type": "multipart/form-data",
      },
    };
    formData.append("files", acceptedFiles[0]);

    axios.defaults.baseURL = "http://api:8080/";
    const response = await axios.post("/files", formData, config);
	const uploadedFile = response.data.files[0].stored;
	setUploadedImage(uploadedFile)
  }, [])

  useEffect(() => {
  	const timerId = setInterval(async () => {

		if (uploadedImage) {
			const processedResult = await axios.get(`files/${uploadedImage}/:result`)
			setProcessedStatus(processedResult.status)
	}
	}, 1000)
	setIntervalId(timerId);
  }, [uploadedImage])

  useEffect(() => {
		if (processedStatus === 200) {
			clearInterval(intervalId)
			setResultFilePath(`files/${uploadedImage}/:result`)
		}
	}, [processedStatus, uploadedImage, intervalId])

  useEffect(() => {
		setPropResultPath(resultFilePath)
	}, [resultFilePath, setPropResultPath])

  const {getRootProps, getInputProps, isDragActive} = useDropzone({onDrop})

  return (
    <div {...getRootProps()}>
      <input {...getInputProps()} />
      {
        isDragActive ?
          <p>Drop the files here ...</p> :
          <p>Drag 'n' drop some files here, or click to select files</p>
      }
    </div>
  )
}

export { MyDropzone };
