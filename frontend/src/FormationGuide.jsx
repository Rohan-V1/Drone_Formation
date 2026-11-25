import { ArrowBackRounded } from "@mui/icons-material";
import { Box, Button, Paper, Typography } from "@mui/material";
import "./FormationGuide.css";

const formations = [
  { name: "V", title: "V formation", shape: "M15 20 L50 70 L85 20", look: "Two arms meet at a point, like the letter V.", advantage: "Keeps the group spread while giving members a clear place relative to the lead." },
  { name: "Inverted V", title: "Inverted V", shape: "M15 70 L50 20 L85 70", look: "An upside-down V, with the point at the top.", advantage: "A compact, symmetric arrangement that keeps members grouped around a lead." },
  { name: "Line/Abreast", title: "Line / abreast", shape: "M15 50 L85 50", look: "Drones sit side by side in one row.", advantage: "Gives a broad view across the direction the group faces." },
  { name: "Column/File", title: "Column / file", shape: "M50 15 L50 85", look: "Drones follow one another in a single line.", advantage: "Uses less width and can keep the group visually compact." },
  { name: "Echelon right", title: "Echelon (right-slanting)", shape: "M18 75 L82 25", look: "A diagonal line that rises toward the right side of the image.", advantage: "Keeps members staggered while the group remains in one line." },
  { name: "Echelon left", title: "Echelon (left-slanting)", shape: "M18 25 L82 75", look: "A diagonal line that falls toward the right side of the image.", advantage: "Keeps members staggered while the group remains in one line." },
  { name: "Diamond", title: "Diamond / rhombus", shape: "M50 12 L84 50 L50 88 L16 50 Z", look: "Four main positions form a diamond around the center.", advantage: "A balanced layout that spreads members around the group center." },
  { name: "Box", title: "Box / square", shape: "M22 22 L78 22 L78 78 L22 78 Z", look: "Positions form the corners of a square or rectangle.", advantage: "Spreads the group in two dimensions and makes its outline easy to see." },
  { name: "Wedge", title: "Wedge", shape: "M15 25 L50 75 L85 25 M32 48 L68 48", look: "A broad, filled V with more than one drone on its arms.", advantage: "Combines a pointed front with a wider group behind it." },
  { name: "Circle", title: "Circle / ring", shape: "M50 14 L76 24 L86 50 L76 76 L50 86 L24 76 L14 50 L24 24 Z", look: "Drones are spaced around a ring-like outline.", advantage: "Surrounds a central area and gives members views from several sides." },
  { name: "Multi-row", title: "Multiple rows", shape: "M18 25 L82 25 M18 50 L82 50 M18 75 L82 75", look: "Two or more roughly parallel rows.", advantage: "Spreads a larger group across more than one line." },
  { name: "Dispersed", title: "Dispersed group", shape: "M20 25 L49 18 L78 34 M28 68 L57 53 L82 78", look: "Drones do not form one clear, repeated pattern.", advantage: "Can give members room to cover different parts of an area." },
];

function matchesFormation(current, item) {
  const value = String(current || "").toLowerCase();
  if (!value) return false;
  if (item.name === "V") return value.includes("v formation") && !value.includes("inverted") && !value.includes("wedge");
  if (item.name === "Inverted V") return value.includes("inverted v");
  if (item.name === "Line/Abreast") return value.includes("line/abreast") || value.includes("horizontal line");
  if (item.name === "Column/File") return value.includes("column/file") || value.includes("vertical line");
  if (item.name === "Echelon right") return value.includes("echelon (right") || value.includes("echelon right");
  if (item.name === "Echelon left") return value.includes("echelon (left") || value.includes("echelon left");
  if (item.name === "Diamond") return value.includes("diamond") || value.includes("rhombus");
  if (item.name === "Box") return value.includes("box") || value.includes("square");
  if (item.name === "Wedge") return value.includes("wedge");
  if (item.name === "Circle") return value.includes("circle") || value.includes("ring");
  if (item.name === "Multi-row") return value.includes("multi-row") || value.includes("multiple rows");
  if (item.name === "Dispersed") return value.includes("dispersed");
  return false;
}

function FormationMark({ shape }) {
  const points = [...shape.matchAll(/[ML]\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)/g)]
    .map((match) => [Number(match[1]), Number(match[2])]);
  return (
    <svg className="formation-mark" viewBox="0 0 100 100" role="img" aria-label="Simple formation diagram">
      <path d={shape} fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      {points.map(([x, y], index) => <circle key={index} cx={x} cy={y} r="4.5" fill="currentColor" stroke="#101722" strokeWidth="1.5" />)}
    </svg>
  );
}

export default function FormationGuide({ currentFormation, onBack }) {
  const selected = formations.find((formation) => matchesFormation(currentFormation, formation));

  return (
    <Box className="formation-guide">
      <Box className="guide-heading">
        <Box>
          <Typography className="guide-kicker">FIELD GUIDE</Typography>
          <Typography variant="h3">Drone formations</Typography>
          <Typography className="guide-lead">A quick guide to the shapes you may see in a swarm image.</Typography>
        </Box>
        <Button variant="outlined" startIcon={<ArrowBackRounded />} onClick={onBack}>Back to image analysis</Button>
      </Box>

      {selected && (
        <Paper className="guide-featured">
          <FormationMark shape={selected.shape} />
          <Box>
            <Typography className="guide-kicker">FROM YOUR LAST RESULT</Typography>
            <Typography variant="h5">{selected.title}</Typography>
            <Typography>{selected.look}</Typography>
            <Typography className="guide-advantage"><strong>Why use it:</strong> {selected.advantage}</Typography>
          </Box>
        </Paper>
      )}

      <Box className="formation-grid">
        {formations.map((formation, index) => (
          <Paper className={`formation-card${selected === formation ? " selected" : ""}`} key={formation.name}>
            <Box className="formation-card-top">
              <span className="formation-number">{String(index + 1).padStart(2, "0")}</span>
              <FormationMark shape={formation.shape} />
            </Box>
            <Typography variant="h6">{formation.title}</Typography>
            <Typography className="formation-look">{formation.look}</Typography>
            <Typography className="formation-advantage"><strong>Possible advantage</strong><br />{formation.advantage}</Typography>
          </Paper>
        ))}
      </Box>

      <Button className="guide-back-bottom" variant="contained" startIcon={<ArrowBackRounded />} onClick={onBack}>Back to image analysis</Button>
    </Box>
  );
}
