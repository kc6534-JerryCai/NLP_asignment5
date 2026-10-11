import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const dataDir = path.join(root, "data", "equity_sentiment", "midterm_basket");
const outputPath = path.join(dataDir, "midterm_basket_results.xlsx");
const fontFamily = "Arial";

async function csvMatrix(fileName) {
  const csvText = await fs.readFile(path.join(dataDir, fileName), "utf8");
  const temp = await Workbook.fromCSV(csvText, { sheetName: "Imported" });
  const used = temp.worksheets.getItemAt(0).getUsedRange();
  return used ? used.values : [];
}

function typedMatrix(matrix) {
  if (!matrix.length) return matrix;
  const headers = matrix[0].map(String);
  const numericNames = new Set([
    "topic_id", "component_count", "N", "minimum_required", "hac_maxlags",
    "bootstrap_block_length", "bootstrap_repetitions", "bootstrap_repetitions_valid",
    "alpha", "beta", "hac_standard_error", "hac_t_value", "raw_p_value",
    "beta_ci_low", "beta_ci_high", "signal_sample_sd", "standardized_effect_bps",
    "standardized_ci_low_bps", "standardized_ci_high_bps", "r_squared",
    "bootstrap_beta_ci_low", "bootstrap_beta_ci_high", "q_value", "selection_beta",
    "selection_sign", "source_min_q", "selection_p", "selection_q_within_signal",
    "selection_t", "N_selection"
  ]);
  const booleanNames = new Set([
    "bootstrap_ci_excludes_zero", "bootstrap_direction_agrees", "fdr_significant",
    "fdr_and_bootstrap_supported"
  ]);
  return matrix.map((row, rowIndex) => row.map((value, colIndex) => {
    if (rowIndex === 0) return String(value);
    const name = headers[colIndex];
    const text = value === null || value === undefined ? "" : String(value).trim();
    if (!text) return null;
    if (numericNames.has(name)) {
      const parsed = Number(text);
      return Number.isFinite(parsed) ? parsed : null;
    }
    if (booleanNames.has(name)) return text.toLowerCase() === "true";
    return text;
  }));
}

function styleDataSheet(sheet, matrix, tableName) {
  if (!matrix.length) return;
  const rows = matrix.length;
  const cols = matrix[0].length;
  const range = sheet.getRangeByIndexes(0, 0, rows, cols);
  range.values = matrix;
  range.format.font = { name: fontFamily, size: 9, color: "#111827" };
  range.format.verticalAlignment = "center";
  const header = sheet.getRangeByIndexes(0, 0, 1, cols);
  header.format = {
    fill: "#1F4E78",
    font: { name: fontFamily, size: 9, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
  };
  header.format.rowHeight = 36;
  range.format.autofitColumns();
  const maxWidths = [16, 34, 18, 28, 22, 22, 20, 15, 16, 16, 16, 16, 16, 18, 18, 24, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18];
  for (let col = 0; col < cols; col++) {
    const current = sheet.getRangeByIndexes(0, col, rows, 1);
    current.format.columnWidth = Math.min(maxWidths[col] ?? 18, 34);
  }
  const headers = matrix[0].map(String);
  for (let col = 0; col < cols; col++) {
    const name = headers[col];
    const body = sheet.getRangeByIndexes(1, col, Math.max(1, rows - 1), 1);
    if (["raw_p_value", "q_value", "beta", "hac_standard_error", "bootstrap_beta_ci_low", "bootstrap_beta_ci_high"].includes(name)) {
      body.format.numberFormat = "0.0000E+00";
    } else if (["standardized_effect_bps", "standardized_ci_low_bps", "standardized_ci_high_bps"].includes(name)) {
      body.format.numberFormat = "0.00";
    } else if (["r_squared"].includes(name)) {
      body.format.numberFormat = "0.000";
    } else if (["N", "component_count", "bootstrap_repetitions_valid", "N_selection"].includes(name)) {
      body.format.numberFormat = "0";
    }
  }
  sheet.tables.add(sheet.getRangeByIndexes(0, 0, rows, cols), true, tableName);
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  sheet.showGridLines = false;
}

const summary = JSON.parse(await fs.readFile(path.join(dataDir, "midterm_basket_summary.json"), "utf8"));
const fullMatrix = typedMatrix(await csvMatrix("midterm_basket_tests.csv"));
const holdoutMatrix = typedMatrix(await csvMatrix("holdout_basket_tests.csv"));
const fullComponents = typedMatrix(await csvMatrix("midterm_basket_components.csv"));
const holdoutComponents = typedMatrix(await csvMatrix("holdout_basket_components.csv"));

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Summary");
const fullSheet = workbook.worksheets.add("Full sample tests");
const holdoutSheet = workbook.worksheets.add("Holdout tests");
const componentSheet = workbook.worksheets.add("Components");

summarySheet.showGridLines = false;
summarySheet.tabColor = "#1F4E78";
fullSheet.tabColor = "#5B9BD5";
holdoutSheet.tabColor = "#70AD47";

summarySheet.getRange("A2:H2").values = [["Mid-term equity basket tests", null, null, null, null, null, null, null]];
summarySheet.getRange("A2:H2").format.font = { name: fontFamily, size: 15, bold: true, color: "#111827" };
summarySheet.getRange("A3:H3").format.borders = { bottom: { style: "thin", color: "#9CA3AF" } };
summarySheet.getRange("A4:B11").values = [
  ["Metric", "Result"],
  ["Full-sample baskets", summary.full_sample_baskets],
  ["Valid full-sample tests", summary.full_sample_valid_tests],
  ["Full-sample FDR significant", summary.full_sample_fdr_significant],
  ["Full-sample FDR + bootstrap", summary.full_sample_fdr_and_bootstrap_supported],
  ["Holdout valid tests", summary.holdout_valid_tests],
  ["Holdout FDR significant", summary.holdout_fdr_significant],
  ["Holdout FDR + bootstrap", summary.holdout_fdr_and_bootstrap_supported],
];
summarySheet.getRange("A4:B4").format = { fill: "#1F4E78", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summarySheet.getRange("A5:B11").format.font = { name: fontFamily, size: 10, color: "#111827" };
summarySheet.getRange("B5:B11").format.numberFormat = "0";
summarySheet.getRange("D4:H11").values = [
  ["Interpretation notes", null, null, null, null],
  ["Full-sample baskets reuse the same-period stock associations used to select constituents.", null, null, null, null],
  ["Their p-values are exploratory and are not independent confirmation.", null, null, null, null],
  [`Early selection ends ${summary.holdout_selection_end}; holdout validation starts ${summary.holdout_validation_start}.`, null, null, null, null],
  ["Directional baskets apply the early or full-sample beta sign and are statistical long/short factors.", null, null, null, null],
  ["Returns are adjusted open-to-close stock returns minus the same-interval SPY return.", null, null, null, null],
  ["Guardian text sentiment is not voter sentiment; association does not establish causality.", null, null, null, null],
  ["Source: Assignment5 equity sentiment outputs and existing yfinance price cache.", null, null, null, null],
];
summarySheet.getRange("D4:H4").format = { fill: "#1F4E78", font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" } };
summarySheet.getRange("D5:H11").format.font = { name: fontFamily, size: 10, color: "#374151" };

const fullHeaders = fullMatrix[0].map(String);
const holdHeaders = holdoutMatrix[0].map(String);
const fullRows = fullMatrix.slice(1).filter(row => row[fullHeaders.indexOf("status")] === "valid" && row[fullHeaders.indexOf("direction_mode")] === "long_only_equal_weight" && row[fullHeaders.indexOf("fdr_significant")] === true).slice(0, 15);
const holdRows = holdoutMatrix.slice(1).filter(row => row[holdHeaders.indexOf("status")] === "valid" && row[holdHeaders.indexOf("fdr_significant")] === true).slice(0, 10);

function selectColumns(headers, rows, names) {
  const indexes = names.map(name => headers.indexOf(name));
  return [names, ...rows.map(row => indexes.map(index => row[index]))];
}

const resultColumns = ["basket_id", "topic_id", "index_type", "component_count", "N", "standardized_effect_bps", "raw_p_value", "q_value", "bootstrap_ci_excludes_zero"];
summarySheet.getRange("A14:I14").values = [["Significant long-only full-sample baskets", null, null, null, null, null, null, null, null]];
summarySheet.getRange("A14:I14").format.font = { name: fontFamily, size: 12, bold: true, color: "#1F4E78" };
const fullSummary = selectColumns(fullHeaders, fullRows, resultColumns);
summarySheet.getRangeByIndexes(15, 0, fullSummary.length, fullSummary[0].length).values = fullSummary;
summarySheet.getRangeByIndexes(15, 0, 1, fullSummary[0].length).format = { fill: "#4472C4", font: { name: fontFamily, size: 9, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", wrapText: true };

const holdStart = 17 + fullRows.length;
summarySheet.getRangeByIndexes(holdStart, 0, 1, 9).values = [["Significant chronological holdout baskets", null, null, null, null, null, null, null, null]];
summarySheet.getRangeByIndexes(holdStart, 0, 1, 9).format.font = { name: fontFamily, size: 12, bold: true, color: "#1F4E78" };
const holdSummary = selectColumns(holdHeaders, holdRows, resultColumns);
summarySheet.getRangeByIndexes(holdStart + 1, 0, holdSummary.length, holdSummary[0].length).values = holdSummary;
summarySheet.getRangeByIndexes(holdStart + 1, 0, 1, holdSummary[0].length).format = { fill: "#70AD47", font: { name: fontFamily, size: 9, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", wrapText: true };

summarySheet.getRange(`F17:F${holdStart + holdSummary.length + 1}`).format.numberFormat = "0.00";
summarySheet.getRange(`G17:H${holdStart + holdSummary.length + 1}`).format.numberFormat = "0.0000E+00";
summarySheet.getRange("A1:I50").format.verticalAlignment = "center";
summarySheet.getRange("A1:I50").format.font.name = fontFamily;
summarySheet.getRange("A1:I50").format.autofitColumns();
summarySheet.getRange("A:A").format.columnWidth = 48;
summarySheet.getRange("D:D").format.columnWidth = 54;
summarySheet.getRange("E:I").format.columnWidth = 18;

styleDataSheet(fullSheet, fullMatrix, "FullSampleTests");
styleDataSheet(holdoutSheet, holdoutMatrix, "HoldoutTests");

const fullCompHeaders = fullComponents[0].map(String);
const holdCompHeaders = holdoutComponents[0].map(String);
const componentHeaders = ["phase", ...fullCompHeaders];
const fullCompRows = fullComponents.slice(1).map(row => ["full_sample", ...row]);
const holdIndex = fullCompHeaders.map(name => holdCompHeaders.indexOf(name));
const holdCompRows = holdoutComponents.slice(1).map(row => ["holdout", ...holdIndex.map(index => index >= 0 ? row[index] : null)]);
styleDataSheet(componentSheet, [componentHeaders, ...fullCompRows, ...holdCompRows], "BasketComponents");

workbook.recalculate();
const summaryInspect = await workbook.inspect({ kind: "table", range: "Summary!A2:I35", include: "values,formulas", tableMaxRows: 35, tableMaxCols: 9 });
console.log(summaryInspect.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
console.log(errors.ndjson);
const preview = await workbook.render({ sheetName: "Summary", range: `A1:I${holdStart + holdSummary.length + 2}`, scale: 1.2, format: "png" });
await fs.writeFile(path.join(dataDir, "midterm_basket_results_preview.png"), new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, preview: path.join(dataDir, "midterm_basket_results_preview.png") }));

