import React from "react";
import {
  Button,
  Header,
  Container,
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableCell,
} from "semantic-ui-react";
import SmartImage from "./SmartImage";

function MeasurementsTable(measurements) {
  return (
    <Table
      style={{ display: "block", overflow: "scroll", marginBottom: "2rem" }}
    >
      <TableHeader>
        <TableRow>
          <TableHeaderCell>Measurement</TableHeaderCell>
          <TableHeaderCell>Value</TableHeaderCell>
        </TableRow>

        {Object.entries(measurements.measurements).map((measurement, index) => {
          let other_measurement = measurement[1].toString();
          return (
            <TableRow key={index}>
              <TableCell>{measurement[0]}</TableCell>
              <TableCell>{other_measurement}</TableCell>
            </TableRow>
          );
        })}
      </TableHeader>
    </Table>
  );
}

export default function SingleImagePredictor() {
  const [imageString, setimageString] = React.useState(null);
  const [modelImage, setModelImage] = React.useState(null);
  const [measurements, setMeasurements] = React.useState(null);

  const fileInputRef = React.useRef();

  const predict = async () => {
    let formData = new FormData();
    formData.append("image", imageString);

    const response = fetch("/api/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Convert the response to an image

        const image = data.image;
        let image_string = `data:image/png;base64,${image}`;
        setModelImage(image_string);

        const measurements = data.measurements;
        console.log(measurements);
        setMeasurements(measurements);
      });
  };

  function handleChange(e) {
    var reader = new FileReader();
    reader.readAsDataURL(e.target.files[0]);
    reader.onloadend = function () {
      setimageString(reader.result);
    };
  }

  return (
    <Container textAlign="center">
      {modelImage ? (
        <div className="content">
          <div style={{ display: "flex" }}>
            <div>
              <Header as="h2">Original Image</Header>
              <SmartImage imageString={imageString} />
            </div>
            <div>
              <Header as="h2">Model Prediction</Header>
              <img src={modelImage} />
            </div>
          </div>
          {measurements ? (
            <MeasurementsTable measurements={measurements} />
          ) : null}
        </div>
      ) : (
        <div className="content">
          <Header as="h1">Analyse Rod Ring Image</Header>
          {imageString ? <SmartImage imageString={imageString} /> : null}
          <Header>
            Upload Rod Ring Image (note: .tif files won't display properly)
          </Header>
          <div>
            <label htmlFor="user-image-file">
              <Button primary onClick={() => fileInputRef.current.click()}>
                Upload
              </Button>
            </label>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleChange}
              hidden
            />
            <Button secondary onClick={predict}>
              Predict
            </Button>
          </div>
        </div>
      )}
    </Container>
  );
}
