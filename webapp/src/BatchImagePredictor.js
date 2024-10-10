import { useState, useRef } from "react";
import {
  Container,
  Header,
  Button,
  Message,
  Transition,
  Loader,
} from "semantic-ui-react";

export default function BatchImagePredictor() {
  const fileInputRef = useRef();
  const [userZip, setUserZip] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleChange(e) {
    var reader = new FileReader();
    reader.readAsDataURL(e.target.files[0]);
    reader.onloadend = function () {
      setUserZip(reader.result);
    };
  }

  const predict = async () => {
    setLoading(true);
    let formData = new FormData();

    formData.append("zip", userZip);

    fetch("/api/batch_predict", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        document.location = url;
        setLoading(false);
      });
  };

  return (
    <Container textAlign="center">
      <Header as="h1">Batch Analyse Ring Rod Images</Header>
      <Transition visible={userZip}>
        <Message positive>
          <Message.Header>File Upload Successfully</Message.Header>
          <p>Press "Predict" to analyse</p>
        </Message>
      </Transition>
      <Header>Upload Ring Rod Images Zip File</Header>
      {loading ? (
        <Loader active inline="centered">
          Analysing Images...
        </Loader>
      ) : (
        <div>
          <label htmlFor="user-image-file">
            <Button primary onClick={() => fileInputRef.current.click()}>
              Upload
            </Button>
          </label>
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip"
            onChange={handleChange}
            hidden
          />
          <Button secondary onClick={predict}>
            Predict
          </Button>
        </div>
      )}
    </Container>
  );
}
