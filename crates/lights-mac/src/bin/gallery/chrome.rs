use lights_mac::Host;
use objc2::rc::Retained;
use objc2_app_kit::{
    NSAutoresizingMaskOptions, NSFont, NSSegmentSwitchTracking, NSSegmentedControl, NSTextField, NSView,
};
use objc2_foundation::{MainThreadMarker, NSArray, NSPoint, NSRect, NSSize, NSString};

pub const CHROME_POINTS: f64 = 92.0;
const MARGIN: f64 = 8.0;
const ROW_POINTS: f64 = 24.0;
const SEGMENT_POINTS: f64 = 96.0;
const TEXT_POINTS: f64 = 12.0;

pub struct Chrome {
    render: Retained<NSView>,
    selector: Retained<NSSegmentedControl>,
    status: Retained<NSTextField>,
    caption: Retained<NSTextField>,
}

impl Chrome {
    pub fn install(host: &Host, mtm: MainThreadMarker, names: &[&str]) -> Result<Self, &'static str> {
        let window = host.window();
        let render = window.contentView().ok_or("the host window has no content view")?;
        let whole = render.frame();
        let (width, height) = (whole.size.width, whole.size.height);
        let container = unsafe { NSView::initWithFrame(mtm.alloc(), whole) };
        container.setWantsLayer(true);
        window.setContentView(Some(&container));
        let pinned_top = NSAutoresizingMaskOptions::NSViewMinYMargin;
        unsafe {
            render.setFrame(NSRect::new(
                NSPoint::new(0.0, 0.0),
                NSSize::new(width, (height - CHROME_POINTS).max(0.0)),
            ));
            render.setAutoresizingMask(
                NSAutoresizingMaskOptions::NSViewWidthSizable | NSAutoresizingMaskOptions::NSViewHeightSizable,
            );
            container.addSubview(&render);
        }

        let labels = NSArray::from_vec(names.iter().map(|n| NSString::from_str(n)).collect());
        let top = height - MARGIN - ROW_POINTS;
        let selector = unsafe {
            NSSegmentedControl::segmentedControlWithLabels_trackingMode_target_action(
                &labels,
                NSSegmentSwitchTracking::SelectOne,
                None,
                None,
                mtm,
            )
        };
        unsafe {
            for index in 0..names.len() {
                selector.setWidth_forSegment(SEGMENT_POINTS, index as isize);
            }
            selector.setSelectedSegment(0);
            selector.setRefusesFirstResponder(true);
            selector.sizeToFit();
            selector.setFrameOrigin(NSPoint::new(MARGIN, top));
            selector.setAutoresizingMask(pinned_top | NSAutoresizingMaskOptions::NSViewMaxXMargin);
            container.addSubview(&selector);
        }

        let font = unsafe { NSFont::systemFontOfSize(TEXT_POINTS) };
        let left = MARGIN + selector.frame().size.width + 2.0 * MARGIN;
        let status = unsafe { NSTextField::labelWithString(&NSString::from_str(""), mtm) };
        let caption = unsafe { NSTextField::wrappingLabelWithString(&NSString::from_str(""), mtm) };
        let strips = [
            (&status, NSRect::new(NSPoint::new(left, top + 2.0), NSSize::new(width - left - MARGIN, ROW_POINTS - 4.0))),
            (
                &caption,
                NSRect::new(
                    NSPoint::new(MARGIN, height - CHROME_POINTS + MARGIN),
                    NSSize::new(width - 2.0 * MARGIN, CHROME_POINTS - ROW_POINTS - 3.0 * MARGIN),
                ),
            ),
        ];
        for (label, frame) in strips {
            unsafe {
                label.setFont(Some(&font));
                label.setFrame(frame);
                label.setAutoresizingMask(pinned_top | NSAutoresizingMaskOptions::NSViewWidthSizable);
                container.addSubview(label);
            }
        }
        window.makeFirstResponder(Some(&render));
        Ok(Self {
            render,
            selector,
            status,
            caption,
        })
    }

    pub fn show(&self, index: usize, caption: &str) {
        unsafe {
            self.selector.setSelectedSegment(index as isize);
            self.caption.setStringValue(&NSString::from_str(caption));
        }
    }

    pub fn selected(&self) -> Option<usize> {
        usize::try_from(unsafe { self.selector.selectedSegment() }).ok()
    }

    pub fn set_status(&self, text: &str) {
        unsafe { self.status.setStringValue(&NSString::from_str(text)) };
        if let Some(window) = self.render.window() {
            unsafe { window.displayIfNeeded() };
        }
    }

    pub fn caption_text(&self) -> String {
        unsafe { self.caption.stringValue() }.to_string()
    }

    pub fn status_text(&self) -> String {
        unsafe { self.status.stringValue() }.to_string()
    }

    pub fn labels(&self) -> Vec<String> {
        let count = unsafe { self.selector.segmentCount() };
        (0..count)
            .map(|i| unsafe { self.selector.labelForSegment(i) }.map_or(String::new(), |s| s.to_string()))
            .collect()
    }

    pub fn choose(&self, index: usize) {
        unsafe { self.selector.setSelectedSegment(index as isize) };
    }

    pub fn lands_on_render(&self, point: NSPoint) -> bool {
        let Some(container) = (unsafe { self.render.superview() }) else {
            return false;
        };
        unsafe { container.hitTest(point) }.is_some_and(|hit| std::ptr::eq(&*hit, &*self.render))
    }

    pub fn caption_rect(&self) -> NSRect {
        self.in_window(&self.caption)
    }

    pub fn render_rect(&self) -> NSRect {
        self.in_window(&self.render)
    }

    fn in_window(&self, view: &NSView) -> NSRect {
        view.convertRect_toView(view.bounds(), None)
    }
}
