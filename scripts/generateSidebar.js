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
function formatPath(filePath) {
    return filePath.replace(/\\/g, '/').replace(/\.md$/, '');
}
function formatText(name) {
    return name.replace(/-/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
}
// 自然排序函数
function naturalSort(a, b) {
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}
function getFiles(dir, basePath = '') {
    let files = fs.readdirSync(dir);
    // 对文件和目录进行自然排序
    files.sort(naturalSort);
    const result = [];
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
        }
        else if (file.endsWith('.md') && file.toLowerCase() !== 'readme.md') {
            result.push({
                text: formatText(path.basename(file, '.md')),
                link: formatPath(`/${relativePath}`)
            });
        }
    });
    return result;
}
function buildSidebar(dir) {
    const sidebar = {};
    let sections = fs.readdirSync(dir);
    // 对顶级目录进行自然排序
    sections.sort(naturalSort);
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
