const toggleBtn = document.getElementById('themeToggle');

// 初始化主题
let savedTheme = localStorage.getItem('theme');
if (!savedTheme) savedTheme = 'light';
document.body.setAttribute('data-theme', savedTheme);
toggleBtn.textContent = savedTheme === 'dark' ? '🌞' : '🌙';

// 切换主题
toggleBtn.addEventListener('click', () => {
    const current = document.body.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    toggleBtn.textContent = next === 'dark' ? '🌞' : '🌙';
});
