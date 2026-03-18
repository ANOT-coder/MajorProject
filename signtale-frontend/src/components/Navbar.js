import { Link, useLocation } from "react-router-dom";
import "./Navbar.css";

const links = ["home", "stories", "translate", "about"];

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-logo">
        <span className="logo-dot" />
        Signtale
      </Link>

      <div className="navbar-links">
        {links.map((page) => (
          <Link
            key={page}
            to={page === "home" ? "/" : `/${page}`}
            className={`nav-link ${location.pathname === (page === "home" ? "/" : `/${page}`) ? "active" : ""}`}
          >
            {page.charAt(0).toUpperCase() + page.slice(1)}
          </Link>
        ))}
      </div>

      {/* <Link to="/translate" className="nav-cta">Try it free →</Link> */}
    </nav>
  );
}