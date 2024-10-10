import React from "react";
import { TIFFViewer } from "react-tiff";

export default function SmartImage({ imageString }) {
  const [isTif, setTif] = React.useState(false);

  React.useEffect(() => {
    if (imageString) {
      const isTif = imageString.includes("tif");
      setTif(isTif);
    }
  }, [imageString]);

  return isTif ? <TIFFViewer tiff={imageString} /> : <img src={imageString} />;
}
