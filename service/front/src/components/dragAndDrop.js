import React, {useCallback, useState, useEffect} from 'react'
import { apiUri } from '../string';
import axios from 'axios'
import {useDropzone} from 'react-dropzone'
import styled from "styled-components";

function MyDropzone({setResultPath: setPropResultPath, setOriginalPath: setPropOriginalPath}) {

  const [uploadedImage, setUploadedImage] = useState("");
  const [processedStatus, setProcessedStatus] = useState(0);
  const [intervalId, setIntervalId] = useState(0);
  const [resultFilePath, setResultFilePath] = useState("");
  const [originalFilePath, setOriginalFilePath] = useState("");

  const onDrop = useCallback(async acceptedFiles => {
    // Do something with the files
    const formData = new FormData();
    const config = {
      header: {
        "content-type": "multipart/form-data",
      },
    };
    formData.append("files", acceptedFiles[0]);
    axios.defaults.baseURL = apiUri;
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
    }, 2000)
  setOriginalFilePath(`files/${uploadedImage}`)
	setIntervalId(timerId);
  }, [uploadedImage])

  useEffect(() => {
		if (processedStatus === 200) {
			clearInterval(intervalId)
			setResultFilePath(`files/${uploadedImage}/:result`)
		}
	}, [processedStatus, uploadedImage, intervalId])

  useEffect(() => {
    console.log(resultFilePath)
		setPropResultPath(resultFilePath)
	}, [resultFilePath, setPropResultPath])

  useEffect(() => {
    console.log(originalFilePath)
		setPropOriginalPath(originalFilePath)
	}, [originalFilePath, setPropOriginalPath])

  const {getRootProps, getInputProps, isDragActive} = useDropzone({onDrop})

  return (
    <Parent {...getRootProps()}>
      <input {...getInputProps()} />
      {
        isDragActive ?
          <p>Drop the files here ...</p> :
          <p>Drag 'n' drop some files here, or click to select files</p>
      }
    </Parent>
  )
}

const Parent = styled.div`
padding-top: 10px;
background: #55efc4;
height: 50px;
`;

export { MyDropzone };
