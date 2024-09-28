// import fs from 'fs';
// import path from 'path';

// import fs from 'node:fs';
// import path from 'node:path';

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// 获取当前模块的文件名和目录名
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const docsPath = path.join(__dirname, '../docs');
const outputFilePath = path.join(__dirname, '../docs/.vitepress/sidebar.ts');



function formatPath(filePath: string): string {
  return filePath.replace(/\\/g, '/').replace(/\.md$/, '');
}

function formatText(name: string): string {
  return name.replace(/-/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
}

function getFiles(dir: string, basePath = ''): any[] {
  const files = fs.readdirSync(dir);
  const result: any[] = [];

  files.forEach(file => {
    const fullPath = path.join(dir, file);
    const relativePath = path.join(basePath, file);

    if (fs.statSync(fullPath).isDirectory()) {
      const items = getFiles(fullPath, relativePath);
      if (items.length > 0) {
        result.push({
          text: formatText(file),
          collapsible: true,
          items 
        });
      }
    } else if (file.endsWith('.md') && file.toLowerCase() !== 'readme.md') {
      result.push({
        text: formatText(path.basename(file, '.md')), 
        link: formatPath(`/${relativePath}`) 
      });
    }
  });

  return result;
}

function buildSidebar(dir: string): { [key: string]: any[] } {
  const sidebar: { [key: string]: any[] } = {};
  const sections = fs.readdirSync(dir);

  sections.forEach(section => {
    const fullPath = path.join(dir, section);
    const stats = fs.statSync(fullPath);

    if (stats.isDirectory()) {
      const items = getFiles(fullPath, section);
      const prefix = `/${section}/`;

      if (items.length > 0) {
        sidebar[prefix] = items;
      }
    }
  });
  return sidebar;
}

const sidebar = buildSidebar(docsPath);

const sidebarConfig = `export default ${JSON.stringify(sidebar, null, 2)};`;

fs.writeFileSync(outputFilePath, sidebarConfig, 'utf-8');
console.log('Sidebar configuration generated successfully.');