/**
 * AIWardrobe Pitch Deck - Interactive Navigation
 */

// State
let currentSlide = 1;
const totalSlides = 10;

// DOM Elements
const slidesContainer = document.getElementById('slidesContainer');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const slideIndicator = document.getElementById('slideIndicator');
const progressBar = document.getElementById('progressBar');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateSlide();
    setupKeyboardNavigation();
    setupTouchNavigation();
    setupSlideAnimations();
});

// Update slide display
function updateSlide() {
    // Update slides
    document.querySelectorAll('.slide').forEach((slide, index) => {
        const slideNum = index + 1;
        slide.classList.remove('active', 'prev');

        if (slideNum === currentSlide) {
            slide.classList.add('active');
        } else if (slideNum < currentSlide) {
            slide.classList.add('prev');
        }
    });

    // Update navigation
    prevBtn.disabled = currentSlide === 1;
    nextBtn.disabled = currentSlide === totalSlides;

    // Update indicator
    slideIndicator.textContent = `${currentSlide} / ${totalSlides}`;

    // Update progress bar
    const progress = (currentSlide / totalSlides) * 100;
    progressBar.style.width = `${progress}%`;

    // Trigger slide-specific animations
    triggerSlideAnimations(currentSlide);
}

// Navigate to next slide
function nextSlide() {
    if (currentSlide < totalSlides) {
        currentSlide++;
        updateSlide();
    }
}

// Navigate to previous slide
function prevSlide() {
    if (currentSlide > 1) {
        currentSlide--;
        updateSlide();
    }
}

// Go to specific slide
function goToSlide(slideNumber) {
    if (slideNumber >= 1 && slideNumber <= totalSlides) {
        currentSlide = slideNumber;
        updateSlide();
    }
}

// Keyboard navigation
function setupKeyboardNavigation() {
    document.addEventListener('keydown', (e) => {
        switch (e.key) {
            case 'ArrowRight':
            case ' ':
            case 'Enter':
                e.preventDefault();
                nextSlide();
                break;
            case 'ArrowLeft':
            case 'Backspace':
                e.preventDefault();
                prevSlide();
                break;
            case 'Home':
                e.preventDefault();
                goToSlide(1);
                break;
            case 'End':
                e.preventDefault();
                goToSlide(totalSlides);
                break;
            default:
                // Number keys 1-9 for quick navigation
                if (e.key >= '1' && e.key <= '9') {
                    const num = parseInt(e.key);
                    if (num <= totalSlides) {
                        goToSlide(num);
                    }
                }
                // 0 goes to slide 10
                if (e.key === '0') {
                    goToSlide(10);
                }
                break;
        }
    });
}

// Touch/swipe navigation
function setupTouchNavigation() {
    let touchStartX = 0;
    let touchEndX = 0;

    slidesContainer.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    }, false);

    slidesContainer.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    }, false);

    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                nextSlide();
            } else {
                prevSlide();
            }
        }
    }
}

// Slide-specific animations
function setupSlideAnimations() {
    // Add intersection observer for scroll-triggered animations
    const observerOptions = {
        threshold: 0.1
    };

    const animateOnVisible = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe elements that should animate on visibility
    document.querySelectorAll('.pain-point, .feature-card, .metric-card, .tier, .milestone').forEach(el => {
        animateOnVisible.observe(el);
    });
}

// Trigger animations for specific slides
function triggerSlideAnimations(slideNum) {
    // Slide 4: Traction chart animation
    if (slideNum === 4) {
        const chartLine = document.querySelector('.chart-line');
        if (chartLine) {
            chartLine.style.strokeDashoffset = '1000';
            setTimeout(() => {
                chartLine.style.strokeDashoffset = '0';
            }, 100);
        }
    }

    // Slide 6: Market pyramid animation
    if (slideNum === 6) {
        const pyramidLevels = document.querySelectorAll('.pyramid-level');
        pyramidLevels.forEach((level, index) => {
            level.style.opacity = '0';
            level.style.transform = 'translateY(20px)';
            setTimeout(() => {
                level.style.transition = 'all 0.5s ease';
                level.style.opacity = '1';
                level.style.transform = 'translateY(0)';
            }, index * 200);
        });
    }

    // Slide 8: Pricing tiers animation
    if (slideNum === 8) {
        const tiers = document.querySelectorAll('.tier');
        tiers.forEach((tier, index) => {
            tier.style.opacity = '0';
            tier.style.transform = 'translateY(30px)';
            setTimeout(() => {
                tier.style.transition = 'all 0.4s ease';
                tier.style.opacity = '1';
                tier.style.transform = tier.classList.contains('featured') ? 'scale(1.02)' : 'translateY(0)';
            }, index * 150);
        });
    }

    // Slide 10: Ask amount animation
    if (slideNum === 10) {
        const amount = document.querySelector('.amount');
        if (amount) {
            animateValue(amount, 0, 500, 1500);
        }
    }
}

// Animate number counting up
function animateValue(element, start, end, duration) {
    const startTime = performance.now();
    const originalText = element.textContent;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease out cubic)
        const easeProgress = 1 - Math.pow(1 - progress, 3);

        const current = Math.floor(start + (end - start) * easeProgress);
        element.textContent = current + 'K';

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Fullscreen toggle (optional)
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

// Add F key for fullscreen
document.addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') {
        toggleFullscreen();
    }
});

// Presenter notes (press N to toggle)
let showNotes = false;
document.addEventListener('keydown', (e) => {
    if (e.key === 'n' || e.key === 'N') {
        showNotes = !showNotes;
        document.body.classList.toggle('show-notes', showNotes);
    }
});

// Auto-advance for presentation mode (press P to toggle)
let autoAdvance = false;
let autoAdvanceInterval = null;

document.addEventListener('keydown', (e) => {
    if (e.key === 'p' || e.key === 'P') {
        autoAdvance = !autoAdvance;

        if (autoAdvance) {
            autoAdvanceInterval = setInterval(() => {
                if (currentSlide < totalSlides) {
                    nextSlide();
                } else {
                    clearInterval(autoAdvanceInterval);
                    autoAdvance = false;
                }
            }, 5000); // 5 seconds per slide
            console.log('Auto-advance ON (5 seconds per slide)');
        } else {
            clearInterval(autoAdvanceInterval);
            console.log('Auto-advance OFF');
        }
    }
});

// Export functions for HTML onclick handlers
window.nextSlide = nextSlide;
window.prevSlide = prevSlide;
window.goToSlide = goToSlide;
