# Sample license clause

Not legal advice — have a lawyer check this before you rely on it, especially the
liability and termination parts. It's written to be pasted into whatever agreement
or quote you already send clients, with the bracketed bits filled in.

---

**Software License**

1. **Scope.** [Vendor name] grants [Client name] a non-exclusive, non-transferable
   license to install and run the Forklift Gate Monitor software ("the Software") on
   **one (1) computer, at one (1) physical site/branch**, for up to **four (4)**
   camera inputs, for so long as this agreement remains in effect.

2. **No copying or relocation.** The Client may not copy, move, or reinstall the
   Software on any other computer, virtual machine, or site without
   [Vendor name]'s prior written consent. The Software is technically bound to the
   computer it is installed on and will not run elsewhere; attempting to bypass,
   disable, or reverse engineer that restriction is a breach of this agreement.

3. **No reverse engineering.** The Client may not decompile, disassemble, or
   otherwise attempt to extract the Software's source code, trained model, or
   underlying logic, except to the extent a law that cannot be waived says
   otherwise.

4. **Ownership.** [Vendor name] retains all ownership of the Software, its source
   code, and the trained detection model. This agreement grants a license to use
   them, not a sale.

5. **Additional sites or cameras.** A separate license (and fee, as agreed) is
   required for each additional site/branch or for more than four camera inputs at
   a site. [Vendor name] will issue the technical files needed for an additional
   install on request.

6. **Termination.** [Vendor name] may disable or decline to extend the Client's
   license if the Client breaches this agreement, including by copying the Software
   to unlicensed hardware. On termination, the Client must stop using the Software
   and delete any copies of it.

---

### Why this matters even with the technical lock in place

The hardware lock (see [SETUP.md](SETUP.md)) stops a casual copy from running
elsewhere. It doesn't stop someone determined enough to patch the binary, and it
can't stop a client from, say, photographing your detection logic off a screen. This
clause is what you'd actually point to - in a demand letter or in court - if that
happens; the technical lock is what makes it inconvenient enough that most people
won't bother trying.
