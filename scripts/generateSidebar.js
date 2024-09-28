// scripts/generateSidebar.js
const fs = require('fs');
const path = require('path');

const docsPath = path.join(__dirname, '../docs').replace(/\\/g, '/');
const outputFilePath = path.join(__dirname, '../docs/.vitepress/sidebar.js').replace(/\\/g, '/');

function getFiles(dir, fileList = []) {
  fs.readdirSync(dir).forEach(file => {
    const filePath = path.join(dir, file).replace(/\\/g, '/'); 
    if (fs.statSync(filePath).isDirectory()) {
      getFiles(filePath, fileList);
    } else if (filePath.endsWith('.md') && !filePath.includes('README.md')) {
      const formattedPath = filePath.replace(docsPath, '').replace(/\.md$/, '');
      fileList.push(formattedPath);
    }
  });
  return fileList;
}

const files = getFiles(docsPath);
const sidebarConfig = `export default ${JSON.stringify(files, null, 2)};`;

fs.writeFileSync(outputFilePath, sidebarConfig, 'utf-8');
console.log('Sidebar configuration generated successfully.');
