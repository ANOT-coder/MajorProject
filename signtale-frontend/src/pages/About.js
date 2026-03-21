import "./About.css";


const cards = [
  {
    title: "Our Mission",
    body:  "We believe every person deserves access to stories. Signtale uses AI to make literature and communication available to the deaf and hard-of-hearing community in a beautiful, visual way.",
    gradient: "linear-gradient(135deg, var(--pink), var(--peach))",
  },
  {
    title: "The Technology",
    body:  "Built on FastAPI, Three.js, and a custom ML pipeline, Signtale converts natural language into 3D skeleton animations — frame by frame, sign by sign.",
    gradient: "linear-gradient(135deg, var(--mint), var(--sky))",
  },
  {
    title: "Academic Context",
    body:  "This project is part of final year project at himalaya college, under the supervision of Er. Ashok G.M. sir .",
    gradient: "linear-gradient(135deg, var(--yellow), var(--peach))",
  },
  {
    title: "Accessibility First",
    body:  "Every design and technical decision is made with accessibility in mind. From color contrast to animation speed, Signtale is built to be clear, calm, and inclusive.",
    gradient: "linear-gradient(135deg, var(--lavender), var(--pink))",
  },
];

// Replace the img paths with your actual image paths.
// Put your photos in src/assets/ and import or reference them here.
// e.g. img: "/src/assets/bidisha.jpg"
const team = [
  {
    name:  "Bidisha Amatya",
    email: "bidisha.amatya@gmail.com",
    img:   "/bidi.png",
    bg:    "linear-gradient(135deg, var(--pink), var(--lavender))",
  },
  {
    name:  "Prasanna Shakya",
    email: "prasanna@example.com",
    img:   "/prasanna.jpg",
    bg:    "linear-gradient(135deg, var(--mint), var(--sky))",
  },
  {
    name:  "Prinska Maharjan",
    email: "prinska@example.com",
    img:   "/prinska.png",
    bg:    "linear-gradient(135deg, var(--yellow), var(--peach))",
  },
  {
    name:  "Sajal Maharjan",
    email: "sajalmaharjan1@gmail.com",
    img:   "/sajal.png",
    bg:    "linear-gradient(135deg, var(--peach), var(--pink))",
  },
];

export default function About() {
  return (
    <main>
      <div className="about-hero">
        <span className="section-tag">ℹ️ About</span>
        <h2 className="section-title">Making stories accessible for everyone</h2>
        <p className="about-intro">
          Signtale is a research-driven project combining NLP, ML, and 3D animation to bridge
          the gap between written language and ASL.
        </p>
      </div>

      <div className="about-grid">
        {cards.map((card) => (
          <div key={card.title} className="about-card" style={{ background: card.gradient }}>
            <h3 className="about-card-title">{card.title}</h3>
            <p className="about-card-body">{card.body}</p>
          </div>
        ))}
      </div>

      <div className="about-team-section">
        <span className="section-tag">👥 The Team</span>
        <h2 className="section-title">Meet the people behind Signtale</h2>
        <div className="about-team-grid">
          {team.map((member) => (
            <div key={member.name} className="about-team-card">
              <div className="about-team-avatar" style={{ background: member.bg }}>
                {member.img
                  ? <img
                      src={member.img}
                      alt={member.name}
                      className="about-team-img"
                      onError={(e) => {
                        // fallback to initials if image fails to load
                        e.currentTarget.style.display = "none";
                        e.currentTarget.nextSibling.style.display = "flex";
                      }}
                    />
                  : null
                }
                {/* fallback initials shown if image is missing or fails */}
                <span
                  className="about-team-initials"
                  style={{ display: member.img ? "none" : "flex" }}
                >
                  {member.name.split(" ").map(n => n[0]).join("")}
                </span>
              </div>
              <h3 className="about-team-name">{member.name}</h3>
              <a
                href={`mailto:${member.email}`}
                className="about-team-email"
              >
                {member.email}
              </a>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}