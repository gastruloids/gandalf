import React from "react";
import { Card, Container, Header } from "semantic-ui-react";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <Container>
      <Header as="h1" textAlign="center">
        Welcome to RingRod-eo! AI Analysis of Rod/Ring Images
      </Header>
      <Card.Group centered>
        <Card color="red" link={true} as={Link} to="/single_image" fluid>
          <Card.Content textAlign="center" header="Single Image Analysis" />
          <Card.Content
            textAlign="left"
            description="Analyse a single Rod Ring scan using Deep Learning."
          />
        </Card>
        <Card color="blue" link={true} as={Link} to="/batch_image" fluid>
          <Card.Content textAlign="center" header="Batch Process Images" />
          <Card.Content
            textAlign="left"
            description="Analyse a Batch of images using Deep Learning algorithms."
          />
        </Card>
      </Card.Group>
    </Container>
  );
}
