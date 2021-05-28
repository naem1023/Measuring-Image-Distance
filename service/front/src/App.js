import {useState, useEffect} from 'react';
import './App.css';
import { MyDropzone } from './components/dragAndDrop';

function App() {

	const [resultPath, setResultPath] = useState("");

	useEffect(() => {
		// TOOD: Do something with result file
		console.log(resultPath)
	}, [resultPath])

  return (
    <div className="App">
		<MyDropzone setResultPath={setResultPath} />
    </div>
  );
}

export default App;
