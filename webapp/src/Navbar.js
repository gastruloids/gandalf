import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav>
      <h1>
        <Link to="/">RingRod-eo</Link>
      </h1>
      <ul id="navbar-items">
        <Link to="/single_image">Single Image</Link>
        <Link to="/batch_image">Batch Image</Link>
      </ul>
    </nav>
  );
}
