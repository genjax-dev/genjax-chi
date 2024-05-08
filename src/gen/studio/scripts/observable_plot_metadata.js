const ts = require('typescript');
const fs = require('fs');
const path = require('path');

function readFiles(fileName) {
    const fs = require('fs');

    if (!fs.existsSync(fileName)) {
        console.error('File does not exist:', fileName);
        return;
    }

    const program = ts.createProgram([fileName], {});
    program.getTypeChecker();
    const entries = {}

    for (const sourceFile of program.getSourceFiles()) {
        if (!sourceFile.isDeclarationFile) continue;
        if (!sourceFile.fileName.includes('@observablehq/plot')) {
            continue;
        }

        ts.forEachChild(sourceFile, visit);
    }

    function visit(node) {
        if (ts.isFunctionDeclaration(node) && node.name) {
            const functionName = node.name.getText();
            const moduleName = node.getSourceFile().fileName;
            const kind = moduleName.split('@observablehq/plot/src/')[1].split(/[/.]/)[0];
            const doc = node.jsDoc && node.jsDoc[0].comment
            entries[functionName] = entry = entries[functionName] || {} 
            entry.kind = kind 
            if (doc) {entry.doc = doc} 

        }

        // Recursively visit child nodes
        ts.forEachChild(node, visit);
    }
    return entries
}

const entries = readFiles(path.join(__dirname, 'node_modules/@observablehq/plot/src/index.d.ts'));
fs.writeFileSync(path.join(__dirname, 'observable_plot_metadata.json'), JSON.stringify(entries, null, 2));
