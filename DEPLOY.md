# How to Deploy Hand Tracker to Your iPad

## Option 1: GitHub Pages (Recommended - Free & Permanent)

This is the best option for a permanent, shareable URL.

### Steps:
1. **Push to GitHub:**
   ```bash
   cd /Users/vo/Documents/HandTracker/Hand-Tracking-Using-Opencv
   git add index.html
   git commit -m "Add web version"
   git push origin main
   ```

2. **Enable GitHub Pages:**
   - Go to your GitHub repository: https://github.com/WillVo1/Hand-Tracking-Using-Opencv
   - Click **Settings** tab
   - Scroll to **Pages** section (left sidebar)
   - Under "Source", select **main** branch
   - Click **Save**

3. **Access on iPad:**
   - Wait 1-2 minutes for deployment
   - Visit: `https://willvo1.github.io/Hand-Tracking-Using-Opencv/index.html`
   - Allow camera access when prompted
   - Works forever, completely free!

---

## Option 2: Netlify (Easiest - 30 seconds)

Super fast drag-and-drop deployment.

### Steps:
1. Go to: https://app.netlify.com/drop
2. Drag the `index.html` file into the box
3. Get instant URL like: `https://your-site.netlify.app`
4. Open that URL on your iPad
5. Done!

**Note:** Free tier gives you permanent hosting. You can create an account to keep it forever.

---

## Option 3: Local Network (Quick Test)

Use this for quick testing on the same WiFi network.

### Steps:
1. **Start server on your Mac:**
   ```bash
   cd /Users/vo/Documents/HandTracker/Hand-Tracking-Using-Opencv
   python -m http.server 8000
   ```

2. **Find your Mac's IP address:**
   - Click Apple menu → System Settings → Network
   - Look for your IP (e.g., 192.168.1.100)

3. **Access on iPad:**
   - Make sure iPad is on same WiFi
   - Open Safari on iPad
   - Go to: `http://YOUR_MAC_IP:8000/index.html`
   - Example: `http://192.168.1.100:8000/index.html`

**Note:** This only works while the server is running on your Mac and both devices are on the same WiFi.

---

## Option 4: Vercel (Alternative to Netlify)

### Steps:
1. Go to: https://vercel.com
2. Sign up/login (free)
3. Click **Add New** → **Project**
4. Import your GitHub repository OR
5. Drag and drop the HTML file
6. Get URL like: `https://your-app.vercel.app`
7. Open on iPad

---

## Recommended: GitHub Pages

For the best experience, I recommend **Option 1 (GitHub Pages)** because:
- ✅ Free forever
- ✅ HTTPS by default (required for camera)
- ✅ Your own domain: `github.io/Hand-Tracking-Using-Opencv`
- ✅ Easy to update (just push changes)
- ✅ Share the link with anyone

## Troubleshooting on iPad

**Camera not working?**
- Make sure you're using **Safari** (best compatibility)
- Allow camera permissions when prompted
- Must be on **HTTPS** (not HTTP) - all options above provide HTTPS
- Try refreshing the page if it doesn't load

**Loading forever?**
- Check your internet connection
- The MediaPipe library is ~10MB, takes a few seconds
- Look for "Loading Hand Tracking Model..." message

**Still issues?**
- Open Safari Developer Console (Settings → Safari → Advanced → Web Inspector)
- Look for error messages
- Make sure iPad is on iOS 14+ for best compatibility
