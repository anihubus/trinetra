import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/globals.css";


function App() {
  return (
    <main>
      <h1>DRISHTI</h1>
      <p>Underwater sonar detection dashboard</p>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);