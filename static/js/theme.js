// Theme Management System
class ThemeManager {
  constructor() {
    this.currentTheme = localStorage.getItem('theme') || 'default';
    this.themes = [
      { id: 'default', name: 'Default', icon: 'fa-palette' },
      { id: 'dark', name: 'Dark', icon: 'fa-moon' },
      { id: 'light', name: 'Light', icon: 'fa-sun' },
      { id: 'ocean', name: 'Ocean', icon: 'fa-water' },
      { id: 'sunset', name: 'Sunset', icon: 'fa-sunset' },
      { id: 'forest', name: 'Forest', icon: 'fa-tree' }
    ];
    this.init();
  }

  init() {
    // Apply saved theme
    this.applyTheme(this.currentTheme);
    
    // Create theme switcher UI
    this.createThemeSwitcher();
    
    // Listen for system theme preference
    if (window.matchMedia) {
      const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
      darkModeQuery.addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
          this.applyTheme(e.matches ? 'dark' : 'default');
        }
      });
    }
  }

  createThemeSwitcher() {
    // Create theme switcher container
    const switcher = document.createElement('div');
    switcher.className = 'theme-switcher';
    switcher.innerHTML = `
      <button class="theme-btn" id="themeToggleBtn">
        <i class="fas fa-palette"></i> Theme
      </button>
      <div class="theme-dropdown" id="themeDropdown">
        ${this.themes.map(theme => `
          <div class="theme-option ${theme.id === this.currentTheme ? 'active' : ''}" 
               data-theme="${theme.id}">
            <span><i class="fas ${theme.icon}"></i> ${theme.name}</span>
            ${theme.id === this.currentTheme ? '<i class="fas fa-check"></i>' : ''}
          </div>
        `).join('')}
      </div>
    `;
    
    document.body.appendChild(switcher);
    
    // Toggle dropdown
    document.getElementById('themeToggleBtn').addEventListener('click', (e) => {
      e.stopPropagation();
      const dropdown = document.getElementById('themeDropdown');
      dropdown.classList.toggle('show');
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!switcher.contains(e.target)) {
        document.getElementById('themeDropdown').classList.remove('show');
      }
    });
    
    // Theme selection
    document.querySelectorAll('.theme-option').forEach(option => {
      option.addEventListener('click', () => {
        const themeId = option.dataset.theme;
        this.setTheme(themeId);
        document.getElementById('themeDropdown').classList.remove('show');
      });
    });
  }

  setTheme(themeId) {
    this.currentTheme = themeId;
    localStorage.setItem('theme', themeId);
    this.applyTheme(themeId);
    this.updateActiveTheme(themeId);
  }

  applyTheme(themeId) {
    document.body.setAttribute('data-theme', themeId);
    
    // Update active theme indicator
    this.updateActiveTheme(themeId);
    
    // Add transition effect
    document.body.style.transition = 'background 0.5s ease';
    
    // Trigger theme change event
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: themeId } }));
  }

  updateActiveTheme(themeId) {
    document.querySelectorAll('.theme-option').forEach(option => {
      if (option.dataset.theme === themeId) {
        option.classList.add('active');
        option.innerHTML = `
          <span><i class="fas ${this.themes.find(t => t.id === themeId).icon}"></i> ${this.themes.find(t => t.id === themeId).name}</span>
          <i class="fas fa-check"></i>
        `;
      } else {
        option.classList.remove('active');
        const theme = this.themes.find(t => t.id === option.dataset.theme);
        option.innerHTML = `
          <span><i class="fas ${theme.icon}"></i> ${theme.name}</span>
        `;
      }
    });
  }
}

// Initialize theme manager when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new ThemeManager();
  });
} else {
  new ThemeManager();
}
