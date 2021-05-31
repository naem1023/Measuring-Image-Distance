import {useState, useEffect} from 'react';
import './App.css';
import { MyDropzone } from './components/dragAndDrop';
import styled, { createGlobalStyle } from "styled-components";

function App() {

	const [resultPath, setResultPath] = useState("");

	useEffect(() => {
		// TOOD: Do something with result file
		console.log(resultPath)
	}, [resultPath])

  return (
    <div className="App">
        <Container>
          <GlobalStyle />
          <Post>
            <Title>원본 이미지 파일</Title>
		    <MyDropzone setResultPath={setResultPath} />
            <Body>원본 이미지</Body>
          </Post>
          <Post>
            <Title>깊이 이미지 파일</Title>
            <Body>깊이 이미지</Body>
          </Post>
          <Post>
            <Title>계산된 미터 값</Title>
            <Body>미터 값</Body>
          </Post>
        </Container>
    </div>
  );
}


const GlobalStyle = createGlobalStyle`
  body {
    margin: 0;
  }
`;

const Container = styled.div`
  min-height: 100vh;
  padding: 280px 0;
  display: grid;
  grid-template-columns: repeat(3, 400px); //박스 가로
  grid-template-rows: repeat(auto-fit, 400px); // 박스 세로
  grid-auto-rows: 500px;
  grid-gap: 50px 20px;
  justify-content: center;
  background: #55efc4;
  box-sizing: border-box;
`;

const Post = styled.div`
  border: 3px solid black;
  border-radius: 20px;
  background: white;
  box-shadow: 10px 5px 5px #7f8fa6;
`;

const Title = styled.div`
  height: 20%;
  display: flex;
  justify-content: center;
  align-items: center;
  border-bottom: 1px solid black;
  font-weight: 600;
`;

const Body = styled.div`
  height: 80%;
  padding: 11px;
  border-radius: 20px;
`;

export default App;
