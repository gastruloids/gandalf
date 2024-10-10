import "semantic-ui-css/semantic.min.css";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import SingleImagePredictor from "./SingleImagePredictor";
import Home from "./Home";
import Navbar from "./Navbar";
import BatchImagePredictor from "./BatchImagePredictor";

export default function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/single_image" element={<SingleImagePredictor />} />
          <Route path="/batch_image" element={<BatchImagePredictor />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}
